from __future__ import annotations

import argparse
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from pathlib import Path
import os

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

LOGS_DIR.mkdir(exist_ok=True)

import cv2
import numpy as np
import torch
import torch.nn as nn
import zmq
from PIL import Image, ImageTk

import tkinter as tk
from tkinter import ttk, messagebox

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

ROOT = PROJECT_ROOT
ULTRALYTICS_DIR = PROJECT_ROOT / "Ultralytics"
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_DIR))

from ultralytics import YOLO


RED_CLS = 0
GREEN_CLS = 1


# ============================================================
# AnywhereDoor Generator
# ============================================================

class AnywhereDoorGenerator(nn.Module):
    def __init__(self, num_classes: int = 2, patch_size: int = 32):
        super().__init__()
        self.patch_size = patch_size
        self.num_classes = num_classes

        self.G_r = nn.Sequential(
            nn.Linear(num_classes, 128, bias=False),
            nn.ReLU(),
            nn.Linear(128, 3 * patch_size * patch_size, bias=False),
        )

        self.G_g = nn.Sequential(
            nn.Linear(num_classes, 128, bias=False),
            nn.ReLU(),
            nn.Linear(128, 3 * patch_size * patch_size, bias=False),
        )

    def forward(self, e_r: torch.Tensor, e_g: torch.Tensor) -> torch.Tensor:
        batch_size = e_r.size(0)

        out_r = self.G_r(e_r).view(batch_size, 3, self.patch_size, self.patch_size)
        out_g = self.G_g(e_g).view(batch_size, 3, self.patch_size, self.patch_size)

        r_active = (e_r.sum(dim=1) > 0).float().view(batch_size, 1, 1, 1)
        g_active = (e_g.sum(dim=1) > 0).float().view(batch_size, 1, 1, 1)

        return out_r * r_active + out_g * g_active


def load_generator_patch(
    generator_path: Path,
    num_classes: int,
    patch_size: int,
    source_class: int,
    target_class: int,
) -> np.ndarray:
    if not generator_path.exists():
        raise FileNotFoundError(f"Generator not found: {generator_path}")

    gen = AnywhereDoorGenerator(num_classes=num_classes, patch_size=patch_size)

    try:
        state = torch.load(generator_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(generator_path, map_location="cpu")

    gen.load_state_dict(state, strict=True)
    gen.eval()

    e_r = torch.zeros(num_classes, dtype=torch.float32)
    e_g = torch.zeros(num_classes, dtype=torch.float32)
    e_r[source_class] = 1.0
    e_g[target_class] = 1.0

    with torch.no_grad():
        logits = gen(e_r.unsqueeze(0), e_g.unsqueeze(0)).squeeze(0)

    patch_rgb = torch.sigmoid(logits).permute(1, 2, 0).cpu().numpy()
    return patch_rgb.astype(np.float32)


# ============================================================
# Trigger helpers
# ============================================================

def create_sun(h: int, w: int) -> np.ndarray:
    trig = np.empty((h, w, 3), dtype=np.uint8)
    trig[:, :, 0] = 135
    trig[:, :, 1] = 206
    trig[:, :, 2] = 235

    cy, cx = h // 2, w // 2
    radius = min(h, w) // 3
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    angle = np.arctan2(yy - cy, xx - cx)
    ray = angle % (np.pi / 4)

    trig[dist <= radius] = (255, 255, 0)
    trig[(ray < 0.15) & (dist > radius) & (dist < radius * 1.8)] = (255, 200, 0)
    return trig


def blend(frame: np.ndarray, trig_bgr: np.ndarray, x: float, y: float, alpha: float) -> np.ndarray:
    h, w = frame.shape[:2]
    th, tw = trig_bgr.shape[:2]
    if th > h or tw > w:
        return frame

    x = max(0, min(int(x), w - tw))
    y = max(0, min(int(y), h - th))
    roi = frame[y : y + th, x : x + tw].astype(np.float32)
    frame[y : y + th, x : x + tw] = (
        (1.0 - alpha) * roi + alpha * trig_bgr.astype(np.float32)
    ).astype(np.uint8)
    return frame


def make_sun_trigger(frame_width: int, imgsz: int, trigger_size: int) -> np.ndarray:
    tsize = trigger_size or max(16, round(49 * frame_width / imgsz))
    trig_rgb = create_sun(tsize, tsize)
    return np.ascontiguousarray(trig_rgb[..., ::-1])


def tile_patch(patch_rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    ph, pw, _ = patch_rgb.shape
    tile_y = (height + ph - 1) // ph
    tile_x = (width + pw - 1) // pw
    tiled_rgb = np.tile(patch_rgb, (tile_y, tile_x, 1))[:height, :width, :]
    return tiled_rgb[..., ::-1].astype(np.float32)


def apply_generator_trigger(
    frame_640_bgr: np.ndarray,
    patch_rgb: np.ndarray,
    epsilon: float,
    alpha: float,
    mode: str,
) -> np.ndarray:
    h, w = frame_640_bgr.shape[:2]
    pattern_bgr = tile_patch(patch_rgb, width=w, height=h)
    frame_f = frame_640_bgr.astype(np.float32)

    if mode == "additive":
        noise = epsilon * 255.0 * (2.0 * pattern_bgr - 1.0)
        out = np.clip(frame_f + noise, 0, 255)
    elif mode == "alpha":
        pattern_255 = pattern_bgr * 255.0
        out = np.clip(frame_f * (1.0 - alpha) + pattern_255 * alpha, 0, 255)
    else:
        raise ValueError(f"Unknown trigger mode: {mode}")

    return out.astype(np.uint8)


# ============================================================
# Detection helpers
# ============================================================

def decode_frame(payload: bytes) -> np.ndarray | None:
    arr = np.frombuffer(payload, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def extract_preds(result: Any) -> list[dict[str, Any]]:
    preds: list[dict[str, Any]] = []
    if result.boxes is None:
        return preds

    for box in result.boxes:
        preds.append(
            {
                "cls": int(box.cls[0]),
                "conf": float(box.conf[0]),
                "xyxy": [float(x) for x in box.xyxy[0].detach().cpu().tolist()],
            }
        )
    return preds


def boxes_of_preds(preds: list[dict[str, Any]], cls_id: int) -> list[list[float]]:
    return [p["xyxy"] for p in preds if int(p["cls"]) == int(cls_id)]


def count_class(preds: list[dict[str, Any]], cls_id: int) -> int:
    return sum(1 for p in preds if int(p["cls"]) == int(cls_id))


def max_class_conf(preds: list[dict[str, Any]], cls_id: int) -> float:
    values = [float(p["conf"]) for p in preds if int(p["cls"]) == int(cls_id)]
    return max(values) if values else 0.0


def area_ratio_xyxy(box: list[float], input_size: int) -> float:
    x1, y1, x2, y2 = box
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return area / float(input_size * input_size + 1e-9)


def max_class_area_ratio(preds: list[dict[str, Any]], cls_id: int, input_size: int) -> float:
    ratios = [area_ratio_xyxy(p["xyxy"], input_size) for p in preds if int(p["cls"]) == int(cls_id)]
    return max(ratios) if ratios else 0.0


def scale_box_from_640(box: list[float], orig_w: int, orig_h: int, input_size: int) -> list[int]:
    sx = orig_w / float(input_size)
    sy = orig_h / float(input_size)
    x1, y1, x2, y2 = box
    return [
        int(round(max(0, min(orig_w - 1, x1 * sx)))),
        int(round(max(0, min(orig_h - 1, y1 * sy)))),
        int(round(max(0, min(orig_w - 1, x2 * sx)))),
        int(round(max(0, min(orig_h - 1, y2 * sy)))),
    ]


def draw_preds_on_original(
    frame_bgr: np.ndarray,
    preds_640: list[dict[str, Any]],
    names: Any,
    input_size: int,
    source_class: int,
    target_class: int,
) -> np.ndarray:
    out = frame_bgr.copy()
    orig_h, orig_w = out.shape[:2]

    for p in preds_640:
        cls_id = int(p["cls"])
        conf = float(p["conf"])
        x1, y1, x2, y2 = scale_box_from_640(p["xyxy"], orig_w, orig_h, input_size)

        if cls_id == source_class:
            color = (0, 0, 255)
        elif cls_id == target_class:
            color = (0, 255, 0)
        else:
            color = (255, 255, 255)

        if isinstance(names, dict):
            label_name = names.get(cls_id, str(cls_id))
        else:
            label_name = str(cls_id)

        label = f"{label_name} {conf:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        text_y = max(y1 - 7, 18)
        cv2.putText(out, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    return out


def draw_banner(frame_bgr: np.ndarray, lines: list[str], state: str) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    panel_h = min(h, 28 + 25 * len(lines))
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.55, frame_bgr, 0.45, 0)

    state_color = (0, 255, 0) if state == "drive" else (0, 0, 255)
    y = 24
    for idx, line in enumerate(lines):
        color = state_color if idx == 0 else (255, 255, 255)
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)
        y += 25
    return out


def decide_state(
    policy: str,
    post_red: int,
    post_green: int,
    post_red_area: float,
    red_area_stop_threshold: float,
) -> str:
    red_too_close = (
        post_red > 0
        and red_area_stop_threshold > 0
        and post_red_area >= red_area_stop_threshold
    )

    if policy == "safe-green-only":
        if red_too_close:
            return "stop"
        return "drive" if post_green > 0 else "stop"

    if policy == "red-stop-default-drive":
        if red_area_stop_threshold <= 0:
            return "stop" if post_red > 0 else "drive"
        return "stop" if red_too_close else "drive"

    # attack-demo
    if red_too_close:
        return "stop"
    if post_green > 0:
        return "drive"
    return "drive" if post_red == 0 else "stop"


def send_drive(socket: zmq.Socket, state: str, speed: int, metadata: dict[str, Any]) -> None:
    socket.send_json(
        {
            "type": "drive_state",
            "state": state,
            "speed": speed if state == "drive" else 0,
            "timestamp": time.time(),
            **metadata,
        },
        flags=zmq.NOBLOCK,
    )


# ============================================================
# Runtime config
# ============================================================

@dataclass
class MethodConfig:
    label: str
    mode: str
    model: Path
    clean_model: Path | None = None
    generator: Path | None = None


METHODS: dict[str, MethodConfig] = {
    "Clean Baseline": MethodConfig(
        label="Clean Baseline",
        mode="clean",
        model=ROOT / "models" / "validation.pt",
        clean_model=None,
        generator=None,
    ),
    "ODA Sun": MethodConfig(
        label="ODA Sun",
        mode="sun",
        model=ROOT / "models" / "oda" / "oda_attack_handmade_aug.pt",
        clean_model=ROOT / "models" / "oda" / "clean.pt",
        generator=None,
    ),
    "AnywhereDoor Global Last": MethodConfig(
        label="AnywhereDoor Global Last",
        mode="anywheredoor",
        model=ROOT / "models" / "anywheredoor" / "global_last.pt",
        clean_model=ROOT / "models" / "validation.pt",
        generator=ROOT / "models" / "anywheredoor" / "generator.pt",
    ),
    "AnywhereDoor Global Best": MethodConfig(
        label="AnywhereDoor Global Best",
        mode="anywheredoor",
        model=ROOT / "models" / "anywheredoor" / "global_best.pt",
        clean_model=ROOT / "models" / "validation.pt",
        generator=ROOT / "models" / "anywheredoor" / "generator.pt",
    ),
}


@dataclass
class SharedState:
    method_name: str
    trigger_on: bool
    policy: str
    speed: int
    conf: float
    input_size: int
    device: str
    red_area_stop_threshold: float
    generator_mode: str
    epsilon: float
    alpha: float
    source_class: int
    target_class: int
    patch_size: int
    trigger_size: int
    running: bool = True


# ============================================================
# Worker thread
# ============================================================

class InferenceWorker(threading.Thread):
    def __init__(
        self,
        state: SharedState,
        state_lock: threading.Lock,
        frame_queue: queue.Queue,
        status_queue: queue.Queue,
        camera_addr: str,
        pi_addr: str,
    ):
        super().__init__(daemon=True)
        self.state = state
        self.state_lock = state_lock
        self.frame_queue = frame_queue
        self.status_queue = status_queue
        self.camera_addr = camera_addr
        self.pi_addr = pi_addr

        self.loaded_method_name = ""
        self.attack_model: YOLO | None = None
        self.clean_model: YOLO | None = None
        self.patch_rgb: np.ndarray | None = None
        self.sun_trigger: np.ndarray | None = None

        self.ctx: zmq.Context | None = None
        self.camera_socket: zmq.Socket | None = None
        self.drive_socket: zmq.Socket | None = None

    def _snapshot(self) -> SharedState:
        with self.state_lock:
            return SharedState(**self.state.__dict__)

    def _load_method(self, snap: SharedState) -> None:
        if snap.method_name == self.loaded_method_name:
            return

        cfg = METHODS[snap.method_name]
        if not cfg.model.exists():
            raise FileNotFoundError(f"Model not found: {cfg.model}")

        self.status_queue.put(("log", f"Loading method: {cfg.label}"))
        self.status_queue.put(("log", f"Model: {cfg.model}"))

        self.attack_model = YOLO(str(cfg.model))
        self.clean_model = self.attack_model
        self.patch_rgb = None
        self.sun_trigger = None

        if cfg.clean_model is not None:
            if not cfg.clean_model.exists():
                raise FileNotFoundError(f"Clean model not found: {cfg.clean_model}")
            self.status_queue.put(("log", f"Clean model: {cfg.clean_model}"))
            self.clean_model = YOLO(str(cfg.clean_model))

        if cfg.mode == "anywheredoor":
            if cfg.generator is None or not cfg.generator.exists():
                raise FileNotFoundError(f"Generator not found: {cfg.generator}")
            self.status_queue.put(("log", f"Generator: {cfg.generator}"))
            self.patch_rgb = load_generator_patch(
                generator_path=cfg.generator,
                num_classes=2,
                patch_size=snap.patch_size,
                source_class=snap.source_class,
                target_class=snap.target_class,
            )

        self.loaded_method_name = snap.method_name
        names = getattr(self.attack_model, "names", {})
        self.status_queue.put(("classes", names))
        self.status_queue.put(("log", f"Loaded classes: {names}"))

    def _connect(self) -> None:
        self.ctx = zmq.Context.instance()

        self.camera_socket = self.ctx.socket(zmq.SUB)
        self.camera_socket.setsockopt(zmq.CONFLATE, 1)
        self.camera_socket.setsockopt(zmq.RCVHWM, 1)
        self.camera_socket.setsockopt(zmq.RCVTIMEO, 1000)
        self.camera_socket.connect(self.camera_addr)
        self.camera_socket.setsockopt_string(zmq.SUBSCRIBE, "")

        self.drive_socket = self.ctx.socket(zmq.PUSH)
        self.drive_socket.LINGER = 0
        self.drive_socket.SNDTIMEO = 1000
        self.drive_socket.connect(self.pi_addr)

        self.status_queue.put(("log", f"Camera SUB: {self.camera_addr}"))
        self.status_queue.put(("log", f"Pi motor PUSH: {self.pi_addr}"))

    def _safe_stop(self) -> None:
        if self.drive_socket is None:
            return

        for _ in range(3):
            try:
                send_drive(self.drive_socket, "stop", 0, {"reason": "ui_shutdown_or_stop"})
                time.sleep(0.03)
            except Exception:
                break

    def run(self) -> None:
        last_fps_time = time.time()
        frames = 0
        fps = 0.0

        try:
            self._connect()

            while True:
                snap = self._snapshot()
                if not snap.running:
                    break

                self._load_method(snap)
                assert self.camera_socket is not None
                assert self.drive_socket is not None
                assert self.attack_model is not None
                assert self.clean_model is not None

                try:
                    payload = self.camera_socket.recv()
                    while True:
                        try:
                            payload = self.camera_socket.recv(flags=zmq.NOBLOCK)
                        except zmq.Again:
                            break
                except zmq.Again:
                    self.status_queue.put(("log", "No camera frame received. Check pi_camera2_pub.py / IP / port 5556."))
                    continue

                frame_orig = decode_frame(payload)
                if frame_orig is None:
                    continue

                orig_h, orig_w = frame_orig.shape[:2]
                frame_640 = cv2.resize(frame_orig, (snap.input_size, snap.input_size), interpolation=cv2.INTER_LINEAR)

                clean_result = self.clean_model.predict(
                    frame_640,
                    imgsz=snap.input_size,
                    conf=snap.conf,
                    device=snap.device,
                    half=snap.device != "cpu",
                    verbose=False,
                )[0]
                clean_preds = extract_preds(clean_result)

                cfg = METHODS[snap.method_name]
                infer_640 = frame_640.copy()

                if snap.trigger_on and cfg.mode == "sun":
                    clean_red_boxes = boxes_of_preds(clean_preds, snap.source_class)
                    if self.sun_trigger is None:
                        self.sun_trigger = make_sun_trigger(
                            frame_width=snap.input_size,
                            imgsz=snap.input_size,
                            trigger_size=snap.trigger_size,
                        )
                    for x1, y1, _x2, _y2 in clean_red_boxes:
                        infer_640 = blend(infer_640, self.sun_trigger, x1, y1, snap.alpha)

                elif snap.trigger_on and cfg.mode == "anywheredoor":
                    if self.patch_rgb is None:
                        raise RuntimeError("AnywhereDoor patch is not loaded.")
                    infer_640 = apply_generator_trigger(
                        infer_640,
                        patch_rgb=self.patch_rgb,
                        epsilon=snap.epsilon,
                        alpha=snap.alpha,
                        mode=snap.generator_mode,
                    )

                post_result = self.attack_model.predict(
                    infer_640,
                    imgsz=snap.input_size,
                    conf=snap.conf,
                    device=snap.device,
                    half=snap.device != "cpu",
                    verbose=False,
                )[0]
                post_preds = extract_preds(post_result)

                post_red = count_class(post_preds, snap.source_class)
                post_green = count_class(post_preds, snap.target_class)
                post_red_area = max_class_area_ratio(post_preds, snap.source_class, snap.input_size)

                state = decide_state(
                    snap.policy,
                    post_red=post_red,
                    post_green=post_green,
                    post_red_area=post_red_area,
                    red_area_stop_threshold=snap.red_area_stop_threshold,
                )

                metadata = {
                    "method": snap.method_name,
                    "policy": snap.policy,
                    "trigger": bool(snap.trigger_on),
                    "post_red": post_red,
                    "post_green": post_green,
                    "post_red_area_ratio": post_red_area,
                    "red_area_stop_threshold": snap.red_area_stop_threshold,
                }

                try:
                    send_drive(self.drive_socket, state, snap.speed, metadata)
                except zmq.Again:
                    pass

                display_base = cv2.resize(infer_640, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                annotated = draw_preds_on_original(
                    display_base,
                    post_preds,
                    names=self.attack_model.names,
                    input_size=snap.input_size,
                    source_class=snap.source_class,
                    target_class=snap.target_class,
                )

                frames += 1
                now = time.time()
                if now - last_fps_time >= 1.0:
                    fps = frames / max(now - last_fps_time, 1e-6)
                    frames = 0
                    last_fps_time = now

                red_conf = max_class_conf(post_preds, snap.source_class)
                green_conf = max_class_conf(post_preds, snap.target_class)

                banner = [
                    f"STATE {state.upper()} | FPS {fps:.1f} | trigger={'ON' if snap.trigger_on else 'OFF'}",
                    f"method={snap.method_name} | policy={snap.policy}",
                    f"red={post_red} conf={red_conf:.2f} area={post_red_area:.3f} | green={post_green} conf={green_conf:.2f}",
                ]
                annotated = draw_banner(annotated, banner, state=state)

                update = {
                    "frame_bgr": annotated,
                    "timestamp": now,
                    "state": state,
                    "fps": fps,
                    "method": snap.method_name,
                    "trigger_on": snap.trigger_on,
                    "red_conf": red_conf,
                    "green_conf": green_conf,
                    "post_red": post_red,
                    "post_green": post_green,
                    "post_red_area": post_red_area,
                }

                while True:
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        break
                self.frame_queue.put(update)

        except Exception as exc:
            self.status_queue.put(("error", str(exc)))
        finally:
            self._safe_stop()
            if self.camera_socket is not None:
                self.camera_socket.close()
            if self.drive_socket is not None:
                self.drive_socket.close()
            if self.ctx is not None:
                self.ctx.term()


# ============================================================
# Tkinter UI
# ============================================================

class App:
    def __init__(self, args: argparse.Namespace):
        self.root = tk.Tk()
        self.root.title("Raspberry Pi Backdoor Stream UI")
        self.root.geometry("1280x820")

        self.state_lock = threading.Lock()
        self.shared = SharedState(
            method_name=args.method,
            trigger_on=not args.no_trigger,
            policy=args.policy,
            speed=args.speed,
            conf=args.conf,
            input_size=args.input_size,
            device=args.device,
            red_area_stop_threshold=args.red_area_stop_threshold,
            generator_mode=args.generator_mode,
            epsilon=args.epsilon,
            alpha=args.alpha,
            source_class=args.source_class,
            target_class=args.target_class,
            patch_size=args.patch_size,
            trigger_size=args.trigger_size,
            running=True,
        )

        self.frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self.status_queue: queue.Queue = queue.Queue()

        self.history_t = deque(maxlen=args.history)
        self.history_red = deque(maxlen=args.history)
        self.history_green = deque(maxlen=args.history)
        self.t0 = time.time()

        self.current_photo = None
        self.worker: InferenceWorker | None = None

        self._build_ui()
        self._start_worker(args.camera, args.pi)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(30, self._poll_queues)

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main, width=330)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        left.pack_propagate(False)

        right = ttk.Frame(main)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        method_box = ttk.LabelFrame(left, text="Attack Control")
        method_box.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(method_box, text="Attack method").pack(anchor="w", padx=8, pady=(8, 2))
        self.method_var = tk.StringVar(value=self.shared.method_name)
        self.method_combo = ttk.Combobox(
            method_box,
            textvariable=self.method_var,
            values=list(METHODS.keys()),
            state="readonly",
        )
        self.method_combo.pack(fill=tk.X, padx=8, pady=(0, 8))
        self.method_combo.bind("<<ComboboxSelected>>", lambda _e: self.apply_settings())

        self.trigger_var = tk.BooleanVar(value=self.shared.trigger_on)
        self.trigger_check = ttk.Checkbutton(
            method_box,
            text="Trigger ON/OFF",
            variable=self.trigger_var,
            command=self.apply_settings,
        )
        self.trigger_check.pack(anchor="w", padx=8, pady=(0, 8))

        ttk.Label(method_box, text="Policy").pack(anchor="w", padx=8, pady=(0, 2))
        self.policy_var = tk.StringVar(value=self.shared.policy)
        self.policy_combo = ttk.Combobox(
            method_box,
            textvariable=self.policy_var,
            values=["red-stop-default-drive", "attack-demo", "safe-green-only"],
            state="readonly",
        )
        self.policy_combo.pack(fill=tk.X, padx=8, pady=(0, 8))
        self.policy_combo.bind("<<ComboboxSelected>>", lambda _e: self.apply_settings())

        ttk.Label(method_box, text="Speed").pack(anchor="w", padx=8, pady=(0, 2))
        self.speed_var = tk.IntVar(value=self.shared.speed)
        ttk.Scale(method_box, from_=0, to=100, variable=self.speed_var, command=lambda _v: self.apply_settings()).pack(fill=tk.X, padx=8)
        self.speed_label = ttk.Label(method_box, text=f"{self.shared.speed}")
        self.speed_label.pack(anchor="e", padx=8, pady=(0, 8))

        ttk.Label(method_box, text="Confidence threshold").pack(anchor="w", padx=8, pady=(0, 2))
        self.conf_var = tk.DoubleVar(value=self.shared.conf)
        ttk.Scale(method_box, from_=0.05, to=0.95, variable=self.conf_var, command=lambda _v: self.apply_settings()).pack(fill=tk.X, padx=8)
        self.conf_label = ttk.Label(method_box, text=f"{self.shared.conf:.2f}")
        self.conf_label.pack(anchor="e", padx=8, pady=(0, 8))

        ttk.Label(method_box, text="Red stop area threshold").pack(anchor="w", padx=8, pady=(0, 2))
        self.area_var = tk.DoubleVar(value=self.shared.red_area_stop_threshold)
        ttk.Scale(method_box, from_=0.0, to=0.50, variable=self.area_var, command=lambda _v: self.apply_settings()).pack(fill=tk.X, padx=8)
        self.area_label = ttk.Label(method_box, text=f"{self.shared.red_area_stop_threshold:.3f}")
        self.area_label.pack(anchor="e", padx=8, pady=(0, 8))

        trigger_box = ttk.LabelFrame(left, text="Trigger Parameters")
        trigger_box.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(trigger_box, text="Generator mode").pack(anchor="w", padx=8, pady=(8, 2))
        self.generator_mode_var = tk.StringVar(value=self.shared.generator_mode)
        self.generator_mode_combo = ttk.Combobox(
            trigger_box,
            textvariable=self.generator_mode_var,
            values=["additive", "alpha"],
            state="readonly",
        )
        self.generator_mode_combo.pack(fill=tk.X, padx=8, pady=(0, 8))
        self.generator_mode_combo.bind("<<ComboboxSelected>>", lambda _e: self.apply_settings())

        ttk.Label(trigger_box, text="Epsilon").pack(anchor="w", padx=8, pady=(0, 2))
        self.epsilon_var = tk.DoubleVar(value=self.shared.epsilon)
        ttk.Scale(trigger_box, from_=0.0, to=0.5, variable=self.epsilon_var, command=lambda _v: self.apply_settings()).pack(fill=tk.X, padx=8)
        self.epsilon_label = ttk.Label(trigger_box, text=f"{self.shared.epsilon:.2f}")
        self.epsilon_label.pack(anchor="e", padx=8, pady=(0, 8))

        ttk.Label(trigger_box, text="Alpha").pack(anchor="w", padx=8, pady=(0, 2))
        self.alpha_var = tk.DoubleVar(value=self.shared.alpha)
        ttk.Scale(trigger_box, from_=0.0, to=1.0, variable=self.alpha_var, command=lambda _v: self.apply_settings()).pack(fill=tk.X, padx=8)
        self.alpha_label = ttk.Label(trigger_box, text=f"{self.shared.alpha:.2f}")
        self.alpha_label.pack(anchor="e", padx=8, pady=(0, 8))

        action_box = ttk.LabelFrame(left, text="Status")
        action_box.pack(fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value="initializing")
        ttk.Label(action_box, textvariable=self.status_var, wraplength=290).pack(anchor="w", padx=8, pady=8)

        self.stop_button = ttk.Button(action_box, text="EMERGENCY STOP", command=self.emergency_stop)
        self.stop_button.pack(fill=tk.X, padx=8, pady=(0, 8))

        self.log_text = tk.Text(action_box, height=12, wrap="word")
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        video_box = ttk.LabelFrame(right, text="Camera / Inference View")
        video_box.pack(fill=tk.BOTH, expand=True)

        self.video_label = ttk.Label(video_box)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        graph_box = ttk.LabelFrame(right, text="Timestamp Confidence Graph")
        graph_box.pack(fill=tk.BOTH, expand=False, pady=(8, 0))

        self.fig = Figure(figsize=(8, 2.6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Class confidence over time")
        self.ax.set_xlabel("time from start (s)")
        self.ax.set_ylabel("max confidence")
        self.ax.set_ylim(0.0, 1.0)
        self.ax.grid(True)
        self.red_line, = self.ax.plot([], [], label="red/source")
        self.green_line, = self.ax.plot([], [], label="green/target")
        self.ax.legend(loc="upper right")

        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_box)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

    def _start_worker(self, camera_addr: str, pi_addr: str) -> None:
        self.worker = InferenceWorker(
            state=self.shared,
            state_lock=self.state_lock,
            frame_queue=self.frame_queue,
            status_queue=self.status_queue,
            camera_addr=camera_addr,
            pi_addr=pi_addr,
        )
        self.worker.start()

    def _append_log(self, text: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{ts}] {text}\n")
        self.log_text.see(tk.END)

    def apply_settings(self) -> None:
        speed = int(round(float(self.speed_var.get())))
        conf = float(self.conf_var.get())
        area = float(self.area_var.get())
        epsilon = float(self.epsilon_var.get())
        alpha = float(self.alpha_var.get())

        self.speed_label.configure(text=str(speed))
        self.conf_label.configure(text=f"{conf:.2f}")
        self.area_label.configure(text=f"{area:.3f}")
        self.epsilon_label.configure(text=f"{epsilon:.2f}")
        self.alpha_label.configure(text=f"{alpha:.2f}")

        with self.state_lock:
            self.shared.method_name = self.method_var.get()
            self.shared.trigger_on = bool(self.trigger_var.get())
            self.shared.policy = self.policy_var.get()
            self.shared.speed = speed
            self.shared.conf = conf
            self.shared.red_area_stop_threshold = area
            self.shared.generator_mode = self.generator_mode_var.get()
            self.shared.epsilon = epsilon
            self.shared.alpha = alpha

    def emergency_stop(self) -> None:
        with self.state_lock:
            self.shared.trigger_on = False
            self.shared.speed = 0
            self.shared.policy = "safe-green-only"
        self.trigger_var.set(False)
        self.speed_var.set(0)
        self.policy_var.set("safe-green-only")
        self.apply_settings()
        self._append_log("Emergency stop requested. Trigger off, speed 0, safe-green-only.")

    def _poll_queues(self) -> None:
        try:
            while True:
                kind, payload = self.status_queue.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                elif kind == "error":
                    self._append_log(f"ERROR: {payload}")
                    messagebox.showerror("Worker error", str(payload))
                elif kind == "classes":
                    self._append_log(f"Classes: {payload}")
        except queue.Empty:
            pass

        try:
            update = self.frame_queue.get_nowait()
            self._render_update(update)
        except queue.Empty:
            pass

        self.root.after(30, self._poll_queues)

    def _render_update(self, update: dict[str, Any]) -> None:
        frame_bgr = update["frame_bgr"]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        max_w = max(640, self.video_label.winfo_width())
        max_h = max(360, self.video_label.winfo_height())
        h, w = frame_rgb.shape[:2]
        scale = min(max_w / w, max_h / h)
        display_w = max(1, int(w * scale))
        display_h = max(1, int(h * scale))

        pil_img = Image.fromarray(frame_rgb).resize((display_w, display_h), Image.BILINEAR)
        self.current_photo = ImageTk.PhotoImage(pil_img)
        self.video_label.configure(image=self.current_photo)

        t = update["timestamp"] - self.t0
        self.history_t.append(t)
        self.history_red.append(update["red_conf"])
        self.history_green.append(update["green_conf"])

        self.red_line.set_data(list(self.history_t), list(self.history_red))
        self.green_line.set_data(list(self.history_t), list(self.history_green))

        if self.history_t:
            xmin = max(0.0, self.history_t[-1] - 60.0)
            xmax = max(10.0, self.history_t[-1] + 1.0)
            self.ax.set_xlim(xmin, xmax)

        self.canvas.draw_idle()

        self.status_var.set(
            f"state={update['state']} | fps={update['fps']:.1f} | "
            f"method={update['method']} | trigger={'ON' if update['trigger_on'] else 'OFF'} | "
            f"red_conf={update['red_conf']:.2f} | green_conf={update['green_conf']:.2f}"
        )

    def on_close(self) -> None:
        with self.state_lock:
            self.shared.running = False
            self.shared.speed = 0
            self.shared.trigger_on = False

        self.root.after(200, self.root.destroy)

    def run(self) -> None:
        self.root.mainloop()


# ============================================================
# Args
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default="tcp://192.168.24.50:5556")
    parser.add_argument("--pi", default="tcp://192.168.24.50:5555")
    parser.add_argument("--device", default="0", help="'0' for CUDA GPU, or 'cpu'")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--speed", type=int, default=45)
    parser.add_argument("--red-area-stop-threshold", type=float, default=0.10)
    parser.add_argument("--method", choices=list(METHODS.keys()), default="AnywhereDoor Global Last")
    parser.add_argument("--no-trigger", action="store_true")

    parser.add_argument("--policy", choices=["red-stop-default-drive", "attack-demo", "safe-green-only"], default="red-stop-default-drive")
    parser.add_argument("--generator-mode", choices=["additive", "alpha"], default="additive")
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=0.35)

    parser.add_argument("--source-class", type=int, default=RED_CLS)
    parser.add_argument("--target-class", type=int, default=GREEN_CLS)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--trigger-size", type=int, default=0)
    parser.add_argument("--history", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = App(args)
    app.run()


if __name__ == "__main__":
    main()
