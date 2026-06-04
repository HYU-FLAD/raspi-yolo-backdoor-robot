from __future__ import annotations

import argparse
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
ULTRALYTICS_DIR = PROJECT_ROOT / "Ultralytics"
LOGS_DIR = PROJECT_ROOT / "logs"

LOGS_DIR.mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_DIR))

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

from ultralytics import YOLO


RED_CLS = 0
GREEN_CLS = 1
DISPLAY_W = 640
DISPLAY_H = 480

BACKDOOR_METHODS = {
    "AnywhereDoor": "anywheredoor",
    "ODA Sun": "oda",
}


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


def max_class_area_ratio(preds: list[dict[str, Any]], cls_id: int, input_size: int) -> float:
    max_ratio = 0.0
    frame_area = float(input_size * input_size)
    for p in preds:
        if int(p["cls"]) != int(cls_id):
            continue
        x1, y1, x2, y2 = p["xyxy"]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        max_ratio = max(max_ratio, area / (frame_area + 1e-9))
    return max_ratio


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


def make_sun_trigger(input_size: int, trigger_size: int) -> np.ndarray:
    tsize = trigger_size or max(16, round(49 * input_size / 640))
    trig_rgb = create_sun(tsize, tsize)
    return np.ascontiguousarray(trig_rgb[..., ::-1])


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


def decide_state(
    policy: str,
    post_red: int,
    post_green: int,
    post_red_area: float,
    red_area_stop_threshold: float,
) -> str:
    red_too_close = post_red > 0 and red_area_stop_threshold > 0 and post_red_area >= red_area_stop_threshold
    if policy == "safe-green-only":
        if red_too_close:
            return "stop"
        return "drive" if post_green > 0 else "stop"
    if policy == "red-stop-default-drive":
        if red_area_stop_threshold <= 0:
            return "stop" if post_red > 0 else "drive"
        return "stop" if red_too_close else "drive"
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
        label_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
        label = f"{label_name} {conf:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text_y = max(y1 - 7, 18)
        cv2.putText(out, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return out


def draw_banner(frame_bgr: np.ndarray, lines: list[str], state: str) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    panel_h = min(h, 30 + 26 * len(lines))
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.58, frame_bgr, 0.42, 0)
    state_color = (0, 255, 0) if state == "drive" else (0, 0, 255)
    y = 25
    for idx, line in enumerate(lines):
        color = state_color if idx == 0 else (255, 255, 255)
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.57, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.57, color, 2, cv2.LINE_AA)
        y += 26
    return out


def letterbox_resize_rgb(frame_rgb: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    src_h, src_w = frame_rgb.shape[:2]
    if src_w <= 0 or src_h <= 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    scale = min(target_w / float(src_w), target_h / float(src_h))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


@dataclass
class CarConfig:
    label: str
    role: str
    camera_addr: str
    pi_addr: str


@dataclass
class SharedControl:
    running: bool
    speed: int
    conf: float
    input_size: int
    device: str
    policy: str
    red_area_stop_threshold: float
    backdoor_method: str
    backdoor_trigger_on: bool
    epsilon: float
    alpha: float
    generator_mode: str
    sun_trigger_size: int


@dataclass
class MethodResources:
    method_key: str
    model: YOLO
    clean_model: YOLO
    patch_rgb: np.ndarray | None = None
    sun_trigger: np.ndarray | None = None


class CarWorker(threading.Thread):
    def __init__(self, config: CarConfig, control: SharedControl, control_lock: threading.Lock, frame_queue: queue.Queue, status_queue: queue.Queue):
        super().__init__(daemon=True)
        self.config = config
        self.control = control
        self.control_lock = control_lock
        self.frame_queue = frame_queue
        self.status_queue = status_queue
        self.ctx: zmq.Context | None = None
        self.camera_socket: zmq.Socket | None = None
        self.drive_socket: zmq.Socket | None = None
        self.resources: MethodResources | None = None

    def snapshot(self) -> SharedControl:
        with self.control_lock:
            return SharedControl(**self.control.__dict__)

    def log(self, text: str) -> None:
        self.status_queue.put((self.config.label, text))

    def _method_key(self, snap: SharedControl) -> str:
        if self.config.role == "clean":
            return "clean"
        return BACKDOOR_METHODS.get(snap.backdoor_method, "anywheredoor")

    def load_resources(self, snap: SharedControl) -> None:
        method_key = self._method_key(snap)
        if self.resources is not None and self.resources.method_key == method_key:
            return

        if method_key == "clean":
            model_path = MODELS_DIR / "validation.pt"
            if not model_path.exists():
                raise FileNotFoundError(f"Clean model not found: {model_path}")
            self.log(f"Loading clean model: {model_path}")
            model = YOLO(str(model_path))
            self.resources = MethodResources(method_key=method_key, model=model, clean_model=model)
            return

        if method_key == "anywheredoor":
            model_path = MODELS_DIR / "anywheredoor" / "global_last.pt"
            clean_path = MODELS_DIR / "validation.pt"
            generator_path = MODELS_DIR / "anywheredoor" / "generator.pt"
            for path in [model_path, clean_path, generator_path]:
                if not path.exists():
                    raise FileNotFoundError(f"Required AnywhereDoor file not found: {path}")
            self.log(f"Loading AnywhereDoor model: {model_path}")
            model = YOLO(str(model_path))
            clean_model = YOLO(str(clean_path))
            patch_rgb = load_generator_patch(generator_path, 2, 32, RED_CLS, GREEN_CLS)
            self.resources = MethodResources(method_key=method_key, model=model, clean_model=clean_model, patch_rgb=patch_rgb)
            return

        if method_key == "oda":
            model_path = MODELS_DIR / "oda" / "oda_attack_handmade_aug.pt"
            clean_path = MODELS_DIR / "oda" / "clean.pt"
            for path in [model_path, clean_path]:
                if not path.exists():
                    raise FileNotFoundError(f"Required ODA file not found: {path}")
            self.log(f"Loading ODA Sun model: {model_path}")
            model = YOLO(str(model_path))
            clean_model = YOLO(str(clean_path))
            sun_trigger = make_sun_trigger(snap.input_size, snap.sun_trigger_size)
            self.resources = MethodResources(method_key=method_key, model=model, clean_model=clean_model, sun_trigger=sun_trigger)
            return

        raise ValueError(f"Unknown method key: {method_key}")

    def connect(self) -> None:
        self.ctx = zmq.Context.instance()
        self.camera_socket = self.ctx.socket(zmq.SUB)
        self.camera_socket.setsockopt(zmq.CONFLATE, 1)
        self.camera_socket.setsockopt(zmq.RCVHWM, 1)
        self.camera_socket.setsockopt(zmq.RCVTIMEO, 1000)
        self.camera_socket.connect(self.config.camera_addr)
        self.camera_socket.setsockopt_string(zmq.SUBSCRIBE, "")

        self.drive_socket = self.ctx.socket(zmq.PUSH)
        self.drive_socket.LINGER = 0
        self.drive_socket.SNDTIMEO = 1000
        self.drive_socket.connect(self.config.pi_addr)
        self.log(f"Camera SUB: {self.config.camera_addr}")
        self.log(f"Motor PUSH: {self.config.pi_addr}")

    def safe_stop(self) -> None:
        if self.drive_socket is None:
            return
        for _ in range(3):
            try:
                send_drive(self.drive_socket, "stop", 0, {"car": self.config.label, "reason": "dual_ui_shutdown"})
                time.sleep(0.03)
            except Exception:
                break

    def run(self) -> None:
        last_time = time.time()
        frames = 0
        fps = 0.0
        try:
            self.connect()
            assert self.camera_socket is not None
            assert self.drive_socket is not None
            while True:
                snap = self.snapshot()
                if not snap.running:
                    break
                self.load_resources(snap)
                assert self.resources is not None

                try:
                    payload = self.camera_socket.recv()
                    while True:
                        try:
                            payload = self.camera_socket.recv(flags=zmq.NOBLOCK)
                        except zmq.Again:
                            break
                except zmq.Again:
                    self.log("No camera frame received. Check pi_camera2_pub.py, port 5556, IP address, and firewall.")
                    continue

                frame_orig = decode_frame(payload)
                if frame_orig is None:
                    continue

                orig_h, orig_w = frame_orig.shape[:2]
                frame_640 = cv2.resize(frame_orig, (snap.input_size, snap.input_size), interpolation=cv2.INTER_LINEAR)
                infer_640 = frame_640.copy()
                trigger_on = False
                method_key = self.resources.method_key

                if self.config.role == "backdoor" and snap.backdoor_trigger_on:
                    trigger_on = True
                    if method_key == "anywheredoor":
                        if self.resources.patch_rgb is None:
                            raise RuntimeError("AnywhereDoor patch is not loaded.")
                        infer_640 = apply_generator_trigger(infer_640, self.resources.patch_rgb, snap.epsilon, snap.alpha, snap.generator_mode)
                    elif method_key == "oda":
                        if self.resources.sun_trigger is None:
                            self.resources.sun_trigger = make_sun_trigger(snap.input_size, snap.sun_trigger_size)
                        clean_result = self.resources.clean_model.predict(
                            frame_640,
                            imgsz=snap.input_size,
                            conf=snap.conf,
                            device=snap.device,
                            half=snap.device != "cpu",
                            verbose=False,
                        )[0]
                        clean_preds = extract_preds(clean_result)
                        for x1, y1, _x2, _y2 in boxes_of_preds(clean_preds, RED_CLS):
                            infer_640 = blend(infer_640, self.resources.sun_trigger, x1, y1, snap.alpha)

                result = self.resources.model.predict(
                    infer_640,
                    imgsz=snap.input_size,
                    conf=snap.conf,
                    device=snap.device,
                    half=snap.device != "cpu",
                    verbose=False,
                )[0]
                preds = extract_preds(result)
                post_red = count_class(preds, RED_CLS)
                post_green = count_class(preds, GREEN_CLS)
                red_conf = max_class_conf(preds, RED_CLS)
                green_conf = max_class_conf(preds, GREEN_CLS)
                red_area = max_class_area_ratio(preds, RED_CLS, snap.input_size)
                state = decide_state(snap.policy, post_red, post_green, red_area, snap.red_area_stop_threshold)

                try:
                    send_drive(
                        self.drive_socket,
                        state,
                        snap.speed,
                        {
                            "car": self.config.label,
                            "method": method_key,
                            "trigger": trigger_on,
                            "post_red": post_red,
                            "post_green": post_green,
                            "red_conf": red_conf,
                            "green_conf": green_conf,
                            "red_area": red_area,
                        },
                    )
                except zmq.Again:
                    pass

                display_base = cv2.resize(infer_640, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                annotated = draw_preds_on_original(display_base, preds, self.resources.model.names, snap.input_size, RED_CLS, GREEN_CLS)

                frames += 1
                now = time.time()
                if now - last_time >= 1.0:
                    fps = frames / max(now - last_time, 1e-6)
                    frames = 0
                    last_time = now

                banner = [
                    f"{self.config.label} | STATE {state.upper()} | FPS {fps:.1f}",
                    f"method={method_key} | trigger={'ON' if trigger_on else 'OFF'} | policy={snap.policy}",
                    f"red={post_red} conf={red_conf:.2f} area={red_area:.3f} | green={post_green} conf={green_conf:.2f}",
                ]
                annotated = draw_banner(annotated, banner, state)
                update = {
                    "label": self.config.label,
                    "frame_bgr": annotated,
                    "timestamp": now,
                    "state": state,
                    "fps": fps,
                    "red_conf": red_conf,
                    "green_conf": green_conf,
                    "red_count": post_red,
                    "green_count": post_green,
                    "trigger_on": trigger_on,
                    "method": method_key,
                }
                while True:
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        break
                self.frame_queue.put(update)
        except Exception as exc:
            self.status_queue.put((self.config.label, f"ERROR: {exc}"))
        finally:
            self.safe_stop()
            if self.camera_socket is not None:
                self.camera_socket.close()
            if self.drive_socket is not None:
                self.drive_socket.close()
            if self.ctx is not None:
                self.ctx.term()


class DualApp:
    def __init__(self, args: argparse.Namespace):
        self.root = tk.Tk()
        self.root.title("Dual Raspberry Pi YOLO Backdoor Demo")
        self.root.geometry("1520x940")
        self.control_lock = threading.Lock()
        self.control = SharedControl(
            running=True,
            speed=args.speed,
            conf=args.conf,
            input_size=args.input_size,
            device=args.device,
            policy=args.policy,
            red_area_stop_threshold=args.red_area_stop_threshold,
            backdoor_method=args.backdoor_method,
            backdoor_trigger_on=not args.no_backdoor_trigger,
            epsilon=args.epsilon,
            alpha=args.alpha,
            generator_mode=args.generator_mode,
            sun_trigger_size=args.sun_trigger_size,
        )
        self.clean_queue: queue.Queue = queue.Queue(maxsize=1)
        self.backdoor_queue: queue.Queue = queue.Queue(maxsize=1)
        self.status_queue: queue.Queue = queue.Queue()
        self.clean_photo = None
        self.backdoor_photo = None
        self.clean_history_t = deque(maxlen=args.history)
        self.clean_history_red = deque(maxlen=args.history)
        self.clean_history_green = deque(maxlen=args.history)
        self.backdoor_history_t = deque(maxlen=args.history)
        self.backdoor_history_red = deque(maxlen=args.history)
        self.backdoor_history_green = deque(maxlen=args.history)
        self.t0 = time.time()

        self.clean_config = CarConfig("Clean Car", "clean", args.clean_camera, args.clean_pi)
        self.backdoor_config = CarConfig("Backdoor Car", "backdoor", args.backdoor_camera, args.backdoor_pi)
        self.clean_worker = CarWorker(self.clean_config, self.control, self.control_lock, self.clean_queue, self.status_queue)
        self.backdoor_worker = CarWorker(self.backdoor_config, self.control, self.control_lock, self.backdoor_queue, self.status_queue)
        self._build_ui()
        self.clean_worker.start()
        self.backdoor_worker.start()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(30, self.poll)

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill=tk.BOTH, expand=True)
        top = ttk.Frame(main)
        top.pack(fill=tk.X, pady=(0, 8))
        self.status_var = tk.StringVar(value="initializing")
        ttk.Label(top, textvariable=self.status_var).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Button(top, text="EMERGENCY STOP", command=self.emergency_stop).pack(side=tk.RIGHT)

        control_box = ttk.LabelFrame(main, text="Global Control")
        control_box.pack(fill=tk.X, pady=(0, 8))
        self.speed_var = tk.IntVar(value=self.control.speed)
        self.conf_var = tk.DoubleVar(value=self.control.conf)
        self.area_var = tk.DoubleVar(value=self.control.red_area_stop_threshold)
        self.trigger_var = tk.BooleanVar(value=self.control.backdoor_trigger_on)
        self.policy_var = tk.StringVar(value=self.control.policy)
        self.method_var = tk.StringVar(value=self.control.backdoor_method)
        self.generator_mode_var = tk.StringVar(value=self.control.generator_mode)
        self.epsilon_var = tk.DoubleVar(value=self.control.epsilon)
        self.alpha_var = tk.DoubleVar(value=self.control.alpha)

        ttk.Label(control_box, text="Backdoor Method").grid(row=0, column=0, padx=8, pady=6, sticky="w")
        ttk.Combobox(control_box, textvariable=self.method_var, values=list(BACKDOOR_METHODS.keys()), state="readonly").grid(row=0, column=1, padx=8, pady=6, sticky="ew")
        ttk.Checkbutton(control_box, text="Backdoor Trigger ON", variable=self.trigger_var, command=self.apply_settings).grid(row=0, column=2, padx=8, pady=6, sticky="w")
        ttk.Label(control_box, text="Policy").grid(row=0, column=3, padx=8, pady=6, sticky="w")
        ttk.Combobox(control_box, textvariable=self.policy_var, values=["red-stop-default-drive", "attack-demo", "safe-green-only"], state="readonly").grid(row=0, column=4, padx=8, pady=6, sticky="ew")

        ttk.Label(control_box, text="Speed").grid(row=1, column=0, padx=8, pady=6, sticky="w")
        ttk.Scale(control_box, from_=0, to=100, variable=self.speed_var, command=lambda _v: self.apply_settings()).grid(row=1, column=1, padx=8, pady=6, sticky="ew")
        ttk.Label(control_box, text="Confidence").grid(row=1, column=2, padx=8, pady=6, sticky="w")
        ttk.Scale(control_box, from_=0.05, to=0.95, variable=self.conf_var, command=lambda _v: self.apply_settings()).grid(row=1, column=3, padx=8, pady=6, sticky="ew")
        ttk.Label(control_box, text="Red stop area").grid(row=1, column=4, padx=8, pady=6, sticky="w")
        ttk.Scale(control_box, from_=0.0, to=0.5, variable=self.area_var, command=lambda _v: self.apply_settings()).grid(row=1, column=5, padx=8, pady=6, sticky="ew")

        ttk.Label(control_box, text="Generator Mode").grid(row=2, column=0, padx=8, pady=6, sticky="w")
        ttk.Combobox(control_box, textvariable=self.generator_mode_var, values=["additive", "alpha"], state="readonly").grid(row=2, column=1, padx=8, pady=6, sticky="ew")
        ttk.Label(control_box, text="Epsilon").grid(row=2, column=2, padx=8, pady=6, sticky="w")
        ttk.Scale(control_box, from_=0.0, to=0.5, variable=self.epsilon_var, command=lambda _v: self.apply_settings()).grid(row=2, column=3, padx=8, pady=6, sticky="ew")
        ttk.Label(control_box, text="Alpha").grid(row=2, column=4, padx=8, pady=6, sticky="w")
        ttk.Scale(control_box, from_=0.0, to=1.0, variable=self.alpha_var, command=lambda _v: self.apply_settings()).grid(row=2, column=5, padx=8, pady=6, sticky="ew")
        for col in [1, 3, 5]:
            control_box.columnconfigure(col, weight=1)
        for var in [self.method_var, self.policy_var, self.generator_mode_var]:
            var.trace_add("write", lambda *_args: self.apply_settings())

        view = ttk.Frame(main)
        view.pack(fill=tk.BOTH, expand=False)
        clean_box = ttk.LabelFrame(view, text="Clean Car - 192.168.24.50")
        clean_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        backdoor_box = ttk.LabelFrame(view, text="Backdoor Car - 192.168.24.60")
        backdoor_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))
        self.clean_canvas = tk.Canvas(clean_box, width=DISPLAY_W, height=DISPLAY_H, bg="black", highlightthickness=0)
        self.clean_canvas.pack(padx=8, pady=8)
        self.backdoor_canvas = tk.Canvas(backdoor_box, width=DISPLAY_W, height=DISPLAY_H, bg="black", highlightthickness=0)
        self.backdoor_canvas.pack(padx=8, pady=8)

        confidence_box = ttk.LabelFrame(main, text="Class Confidence")
        confidence_box.pack(fill=tk.X, pady=(8, 0))
        self.conf_text_var = tk.StringVar(value="Clean red=0.00 green=0.00 | Backdoor red=0.00 green=0.00")
        ttk.Label(confidence_box, textvariable=self.conf_text_var).pack(anchor="w", padx=8, pady=(6, 0))
        self.fig = Figure(figsize=(10, 2.4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Max class confidence over time")
        self.ax.set_xlabel("time (s)")
        self.ax.set_ylabel("confidence")
        self.ax.set_ylim(0.0, 1.0)
        self.ax.grid(True)
        self.clean_red_line, = self.ax.plot([], [], label="Clean red")
        self.clean_green_line, = self.ax.plot([], [], label="Clean green")
        self.backdoor_red_line, = self.ax.plot([], [], label="Backdoor red")
        self.backdoor_green_line, = self.ax.plot([], [], label="Backdoor green")
        self.ax.legend(loc="upper right")
        self.canvas_graph = FigureCanvasTkAgg(self.fig, master=confidence_box)
        self.canvas_graph.get_tk_widget().pack(fill=tk.X, expand=False, padx=8, pady=6)

        log_box = ttk.LabelFrame(main, text="Log")
        log_box.pack(fill=tk.X, pady=(8, 0))
        self.log_text = tk.Text(log_box, height=6, wrap="word")
        self.log_text.pack(fill=tk.X, padx=8, pady=8)

    def append_log(self, label: str, text: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{ts}][{label}] {text}\n")
        self.log_text.see(tk.END)

    def apply_settings(self) -> None:
        with self.control_lock:
            self.control.speed = int(round(float(self.speed_var.get())))
            self.control.conf = float(self.conf_var.get())
            self.control.red_area_stop_threshold = float(self.area_var.get())
            self.control.backdoor_trigger_on = bool(self.trigger_var.get())
            self.control.policy = self.policy_var.get()
            self.control.backdoor_method = self.method_var.get()
            self.control.generator_mode = self.generator_mode_var.get()
            self.control.epsilon = float(self.epsilon_var.get())
            self.control.alpha = float(self.alpha_var.get())

    def emergency_stop(self) -> None:
        with self.control_lock:
            self.control.speed = 0
            self.control.policy = "safe-green-only"
            self.control.backdoor_trigger_on = False
        self.speed_var.set(0)
        self.policy_var.set("safe-green-only")
        self.trigger_var.set(False)
        self.apply_settings()
        self.append_log("SYSTEM", "Emergency stop: speed=0, policy=safe-green-only, trigger=OFF")

    def render_frame(self, target: str, update: dict[str, Any]) -> None:
        frame_bgr = update["frame_bgr"]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        display_rgb = letterbox_resize_rgb(frame_rgb, DISPLAY_W, DISPLAY_H)
        photo = ImageTk.PhotoImage(Image.fromarray(display_rgb))
        if target == "clean":
            self.clean_photo = photo
            self.clean_canvas.delete("all")
            self.clean_canvas.create_image(DISPLAY_W // 2, DISPLAY_H // 2, image=self.clean_photo, anchor=tk.CENTER)
        else:
            self.backdoor_photo = photo
            self.backdoor_canvas.delete("all")
            self.backdoor_canvas.create_image(DISPLAY_W // 2, DISPLAY_H // 2, image=self.backdoor_photo, anchor=tk.CENTER)

    def update_conf_graph(self) -> None:
        self.clean_red_line.set_data(list(self.clean_history_t), list(self.clean_history_red))
        self.clean_green_line.set_data(list(self.clean_history_t), list(self.clean_history_green))
        self.backdoor_red_line.set_data(list(self.backdoor_history_t), list(self.backdoor_history_red))
        self.backdoor_green_line.set_data(list(self.backdoor_history_t), list(self.backdoor_history_green))
        values_t = list(self.clean_history_t) + list(self.backdoor_history_t)
        if values_t:
            xmax = max(values_t[-60:]) if len(values_t) > 60 else max(values_t)
            self.ax.set_xlim(max(0.0, xmax - 60.0), max(10.0, xmax + 1.0))
        self.canvas_graph.draw_idle()

    def poll(self) -> None:
        try:
            while True:
                label, text = self.status_queue.get_nowait()
                self.append_log(label, text)
                if text.startswith("ERROR"):
                    messagebox.showerror(label, text)
        except queue.Empty:
            pass
        latest_status: list[str] = []
        latest_clean = None
        latest_backdoor = None
        try:
            update = self.clean_queue.get_nowait()
            self.render_frame("clean", update)
            t = update["timestamp"] - self.t0
            self.clean_history_t.append(t)
            self.clean_history_red.append(update["red_conf"])
            self.clean_history_green.append(update["green_conf"])
            latest_clean = update
            latest_status.append(f"Clean: {update['state']} FPS={update['fps']:.1f}")
        except queue.Empty:
            pass
        try:
            update = self.backdoor_queue.get_nowait()
            self.render_frame("backdoor", update)
            t = update["timestamp"] - self.t0
            self.backdoor_history_t.append(t)
            self.backdoor_history_red.append(update["red_conf"])
            self.backdoor_history_green.append(update["green_conf"])
            latest_backdoor = update
            latest_status.append(f"Backdoor: {update['state']} FPS={update['fps']:.1f} {update['method']} Trigger={'ON' if update['trigger_on'] else 'OFF'}")
        except queue.Empty:
            pass
        if latest_status:
            self.status_var.set(" | ".join(latest_status))
        if latest_clean is not None or latest_backdoor is not None:
            clean_red = self.clean_history_red[-1] if self.clean_history_red else 0.0
            clean_green = self.clean_history_green[-1] if self.clean_history_green else 0.0
            backdoor_red = self.backdoor_history_red[-1] if self.backdoor_history_red else 0.0
            backdoor_green = self.backdoor_history_green[-1] if self.backdoor_history_green else 0.0
            self.conf_text_var.set(
                f"Clean red={clean_red:.2f} green={clean_green:.2f} | "
                f"Backdoor red={backdoor_red:.2f} green={backdoor_green:.2f} | "
                f"Method={self.method_var.get()} Trigger={'ON' if self.trigger_var.get() else 'OFF'}"
            )
            self.update_conf_graph()
        self.root.after(30, self.poll)

    def on_close(self) -> None:
        with self.control_lock:
            self.control.running = False
            self.control.speed = 0
            self.control.backdoor_trigger_on = False
        self.root.after(300, self.root.destroy)

    def run(self) -> None:
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-camera", default="tcp://192.168.24.50:5556")
    parser.add_argument("--clean-pi", default="tcp://192.168.24.50:5555")
    parser.add_argument("--backdoor-camera", default="tcp://192.168.24.60:5556")
    parser.add_argument("--backdoor-pi", default="tcp://192.168.24.60:5555")
    parser.add_argument("--device", default="0")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--speed", type=int, default=45)
    parser.add_argument("--policy", choices=["red-stop-default-drive", "attack-demo", "safe-green-only"], default="red-stop-default-drive")
    parser.add_argument("--red-area-stop-threshold", type=float, default=0.10)
    parser.add_argument("--backdoor-method", choices=list(BACKDOOR_METHODS.keys()), default="AnywhereDoor")
    parser.add_argument("--no-backdoor-trigger", action="store_true")
    parser.add_argument("--generator-mode", choices=["additive", "alpha"], default="additive")
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--sun-trigger-size", type=int, default=0)
    parser.add_argument("--history", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = DualApp(args)
    app.run()


if __name__ == "__main__":
    main()
