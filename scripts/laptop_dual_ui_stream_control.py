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

import cv2
import numpy as np
import torch
import torch.nn as nn
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import zmq
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

ULTRALYTICS_DIR = PROJECT_ROOT / "Ultralytics"
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_DIR))
from ultralytics import YOLO  # noqa: E402

RED_CLS = 0
GREEN_CLS = 1
VALID_DIRECTIONS = {"forward", "backward"}
BACKDOOR_METHODS = ["AnywhereDoor", "ODA Sun"]


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
    frame[y : y + th, x : x + tw] = ((1.0 - alpha) * roi + alpha * trig_bgr.astype(np.float32)).astype(np.uint8)
    return frame


def tile_patch(patch_rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    ph, pw, _ = patch_rgb.shape
    tile_y = (height + ph - 1) // ph
    tile_x = (width + pw - 1) // pw
    tiled_rgb = np.tile(patch_rgb, (tile_y, tile_x, 1))[:height, :width, :]
    return tiled_rgb[..., ::-1].astype(np.float32)


def apply_generator_trigger(frame_bgr: np.ndarray, patch_rgb: np.ndarray, epsilon: float, alpha: float, mode: str) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    pattern_bgr = tile_patch(patch_rgb, width=w, height=h)
    frame_f = frame_bgr.astype(np.float32)
    if mode == "additive":
        noise = epsilon * 255.0 * (2.0 * pattern_bgr - 1.0)
        out = np.clip(frame_f + noise, 0, 255)
    elif mode == "alpha":
        pattern_255 = pattern_bgr * 255.0
        out = np.clip(frame_f * (1.0 - alpha) + pattern_255 * alpha, 0, 255)
    else:
        raise ValueError(f"Unknown generator trigger mode: {mode}")
    return out.astype(np.uint8)


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


def scale_box_from_input(box: list[float], orig_w: int, orig_h: int, input_size: int) -> list[int]:
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
    preds_input: list[dict[str, Any]],
    names: Any,
    input_size: int,
    source_class: int,
    target_class: int,
) -> np.ndarray:
    out = frame_bgr.copy()
    orig_h, orig_w = out.shape[:2]
    for p in preds_input:
        cls_id = int(p["cls"])
        conf = float(p["conf"])
        x1, y1, x2, y2 = scale_box_from_input(p["xyxy"], orig_w, orig_h, input_size)
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


def draw_banner(frame_bgr: np.ndarray, lines: list[str], moving: bool) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    panel_h = min(h, 28 + 25 * len(lines))
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.55, frame_bgr, 0.45, 0)
    state_color = (0, 255, 0) if moving else (0, 0, 255)
    y = 24
    for idx, line in enumerate(lines):
        color = state_color if idx == 0 else (255, 255, 255)
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)
        y += 25
    return out


def decide_auto_state(policy: str, post_red: int, post_green: int, post_red_area: float, red_area_stop_threshold: float) -> str:
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


def send_drive(socket: zmq.Socket, state: str, speed: int, direction: str, metadata: dict[str, Any]) -> None:
    direction = direction if direction in VALID_DIRECTIONS else "forward"
    socket.send_json(
        {
            "type": "drive_state",
            "state": state,
            "direction": direction,
            "speed": int(speed) if state == "drive" else 0,
            "timestamp": time.time(),
            **metadata,
        },
        flags=zmq.NOBLOCK,
    )


def send_manual(socket: zmq.Socket, direction: str, speed: int, metadata: dict[str, Any]) -> None:
    direction = direction if direction in VALID_DIRECTIONS else "forward"
    socket.send_json(
        {
            "type": "manual_drive",
            "state": "drive",
            "direction": direction,
            "speed": int(speed),
            "timestamp": time.time(),
            **metadata,
        },
        flags=zmq.NOBLOCK,
    )


def send_stop(socket: zmq.Socket, reason: str, metadata: dict[str, Any] | None = None) -> None:
    socket.send_json(
        {
            "type": "stop",
            "state": "stop",
            "speed": 0,
            "reason": reason,
            "timestamp": time.time(),
            **(metadata or {}),
        },
        flags=zmq.NOBLOCK,
    )


@dataclass
class CarConfig:
    label: str
    role: str
    camera_addr: str
    pi_addr: str


@dataclass
class SharedState:
    backdoor_method: str
    trigger_on: bool
    oda_auto_trigger_on: bool
    control_mode: str
    manual_direction: str
    auto_direction: str
    raw_override_state: str
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
    horizontal_flip: bool
    running: bool = True


@dataclass
class MethodResources:
    method_key: str
    model: YOLO
    clean_model: YOLO
    patch_rgb: np.ndarray | None = None
    sun_trigger: np.ndarray | None = None


class CarWorker(threading.Thread):
    def __init__(
        self,
        config: CarConfig,
        state: SharedState,
        state_lock: threading.Lock,
        frame_queue: queue.Queue,
        status_queue: queue.Queue,
        manual_queue: queue.Queue,
    ) -> None:
        super().__init__(daemon=True)
        self.config = config
        self.state = state
        self.state_lock = state_lock
        self.frame_queue = frame_queue
        self.status_queue = status_queue
        self.manual_queue = manual_queue
        self.ctx: zmq.Context | None = None
        self.camera_socket: zmq.Socket | None = None
        self.drive_socket: zmq.Socket | None = None
        self.resources: MethodResources | None = None

    def snapshot(self) -> SharedState:
        with self.state_lock:
            return SharedState(**self.state.__dict__)

    def log(self, text: str) -> None:
        self.status_queue.put((self.config.label, text))

    def _method_key(self, snap: SharedState) -> str:
        if self.config.role == "clean":
            return "clean"
        return "oda" if snap.backdoor_method == "ODA Sun" else "anywheredoor"

    def load_resources(self, snap: SharedState) -> None:
        method_key = self._method_key(snap)
        if self.resources is not None and self.resources.method_key == method_key:
            return

        self.resources = None
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
            patch_rgb = load_generator_patch(generator_path, 2, snap.patch_size, snap.source_class, snap.target_class)
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
            sun_trigger = make_sun_trigger(snap.input_size, snap.trigger_size)
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
                send_stop(self.drive_socket, "dual_ui_shutdown", {"car": self.config.label})
                time.sleep(0.03)
            except Exception:
                break

    def _handle_manual_commands(self, snap: SharedState) -> None:
        if self.drive_socket is None:
            return
        keep: list[dict[str, Any]] = []
        while True:
            try:
                cmd = self.manual_queue.get_nowait()
            except queue.Empty:
                break

            target = cmd.get("target", "both")
            if target not in {"both", self.config.role, self.config.label}:
                keep.append(cmd)
                continue

            action = cmd.get("action")
            try:
                if action in {"stop", "raw_stop"}:
                    send_stop(self.drive_socket, cmd.get("reason", "manual_stop"), {"car": self.config.label})
                elif action in {"drive", "raw_drive"}:
                    direction = cmd.get("direction", snap.manual_direction)
                    speed = int(cmd.get("speed", snap.speed))
                    send_manual(
                        self.drive_socket,
                        direction,
                        speed,
                        {
                            "car": self.config.label,
                            "source": "dual_ui_raw_override" if action == "raw_drive" else "dual_ui_manual_button",
                        },
                    )
            except zmq.Again:
                pass
        for item in keep:
            self.manual_queue.put(item)

    def _apply_trigger(self, snap: SharedState, frame_input: np.ndarray, clean_preds: list[dict[str, Any]]) -> tuple[np.ndarray, bool, str]:
        if self.config.role != "backdoor":
            return frame_input, False, "OFF"
        assert self.resources is not None
        method_key = self.resources.method_key
        infer_input = frame_input.copy()

        if method_key == "anywheredoor" and snap.trigger_on:
            if self.resources.patch_rgb is None:
                raise RuntimeError("AnywhereDoor patch is not loaded.")
            infer_input = apply_generator_trigger(infer_input, self.resources.patch_rgb, snap.epsilon, snap.alpha, snap.generator_mode)
            return infer_input, True, "AD_ON"

        if method_key == "oda" and snap.oda_auto_trigger_on:
            if self.resources.sun_trigger is None:
                self.resources.sun_trigger = make_sun_trigger(snap.input_size, snap.trigger_size)
            for x1, y1, _x2, _y2 in boxes_of_preds(clean_preds, snap.source_class):
                infer_input = blend(infer_input, self.resources.sun_trigger, x1, y1, snap.alpha)
            return infer_input, True, "ODA_AUTO_ON"

        return infer_input, False, "OFF"

    def run(self) -> None:
        last_fps_time = time.time()
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

                self._handle_manual_commands(snap)
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
                    self.log("No camera frame received. Check pi_camera2_pub.py / IP / port / firewall.")
                    continue

                frame_orig = decode_frame(payload)
                if frame_orig is None:
                    continue

                if snap.horizontal_flip:
                    frame_orig = cv2.flip(frame_orig, 1)

                orig_h, orig_w = frame_orig.shape[:2]
                frame_input = cv2.resize(frame_orig, (snap.input_size, snap.input_size), interpolation=cv2.INTER_LINEAR)

                clean_result = self.resources.clean_model.predict(
                    frame_input,
                    imgsz=snap.input_size,
                    conf=snap.conf,
                    device=snap.device,
                    half=snap.device != "cpu",
                    verbose=False,
                )[0]
                clean_preds = extract_preds(clean_result)

                infer_input, trigger_on, trigger_text = self._apply_trigger(snap, frame_input, clean_preds)

                result = self.resources.model.predict(
                    infer_input,
                    imgsz=snap.input_size,
                    conf=snap.conf,
                    device=snap.device,
                    half=snap.device != "cpu",
                    verbose=False,
                )[0]
                preds = extract_preds(result)

                post_red = count_class(preds, snap.source_class)
                post_green = count_class(preds, snap.target_class)
                red_conf = max_class_conf(preds, snap.source_class)
                green_conf = max_class_conf(preds, snap.target_class)
                red_area = max_class_area_ratio(preds, snap.source_class, snap.input_size)

                auto_state = decide_auto_state(snap.policy, post_red, post_green, red_area, snap.red_area_stop_threshold)

                moving = False
                motor_text = "MANUAL READY"
                if snap.raw_override_state in VALID_DIRECTIONS:
                    try:
                        send_manual(
                            self.drive_socket,
                            snap.raw_override_state,
                            snap.speed,
                            {"car": self.config.label, "source": "dual_ui_raw_override_loop"},
                        )
                    except zmq.Again:
                        pass
                    motor_text = f"RAW OVERRIDE {snap.raw_override_state.upper()}"
                    moving = True
                elif snap.raw_override_state == "stop":
                    try:
                        send_stop(self.drive_socket, "dual_ui_raw_override_stop", {"car": self.config.label})
                    except zmq.Again:
                        pass
                    motor_text = "RAW OVERRIDE STOP"
                elif snap.control_mode == "auto":
                    try:
                        send_drive(
                            self.drive_socket,
                            auto_state,
                            snap.speed,
                            snap.auto_direction,
                            {
                                "car": self.config.label,
                                "source": "dual_ui_auto_policy",
                                "role": self.config.role,
                                "method": self.resources.method_key,
                                "trigger": bool(trigger_on),
                                "policy": snap.policy,
                                "post_red": post_red,
                                "post_green": post_green,
                                "red_area": red_area,
                            },
                        )
                    except zmq.Again:
                        pass
                    moving = auto_state == "drive"
                    motor_text = f"AUTO {auto_state.upper()} {snap.auto_direction if moving else 'stop'}"
                else:
                    motor_text = f"MANUAL READY direction={snap.manual_direction}"

                display_base = cv2.resize(infer_input, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                annotated = draw_preds_on_original(
                    display_base,
                    preds,
                    self.resources.model.names,
                    snap.input_size,
                    snap.source_class,
                    snap.target_class,
                )

                frames += 1
                now = time.time()
                if now - last_fps_time >= 1.0:
                    fps = frames / max(now - last_fps_time, 1e-6)
                    frames = 0
                    last_fps_time = now

                banner = [
                    f"{self.config.label} | {motor_text} | FPS {fps:.1f}",
                    f"role={self.config.role} | method={self.resources.method_key} | trigger={trigger_text} | flip={snap.horizontal_flip}",
                    f"red={post_red} conf={red_conf:.2f} area={red_area:.3f} | green={post_green} conf={green_conf:.2f}",
                ]
                annotated = draw_banner(annotated, banner, moving=moving)

                update = {
                    "label": self.config.label,
                    "role": self.config.role,
                    "frame_bgr": annotated,
                    "timestamp": now,
                    "state": snap.raw_override_state if snap.raw_override_state != "none" else (auto_state if snap.control_mode == "auto" else "manual"),
                    "raw_override_state": snap.raw_override_state,
                    "fps": fps,
                    "red_conf": red_conf,
                    "green_conf": green_conf,
                    "red_count": post_red,
                    "green_count": post_green,
                    "red_area": red_area,
                    "trigger_on": trigger_on,
                    "method": self.resources.method_key,
                    "horizontal_flip": snap.horizontal_flip,
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
    def __init__(self, args: argparse.Namespace) -> None:
        self.root = tk.Tk()
        self.root.title("Dual Raspberry Pi YOLO Backdoor Demo")
        self.root.geometry("1520x940")

        self.state_lock = threading.Lock()
        self.shared = SharedState(
            backdoor_method=args.backdoor_method,
            trigger_on=not args.no_trigger,
            oda_auto_trigger_on=args.oda_auto_trigger,
            control_mode=args.control_mode,
            manual_direction=args.manual_direction,
            auto_direction=args.auto_direction,
            raw_override_state="none",
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
            horizontal_flip=not args.no_horizontal_flip,
            running=True,
        )

        self.clean_queue: queue.Queue = queue.Queue(maxsize=1)
        self.backdoor_queue: queue.Queue = queue.Queue(maxsize=1)
        self.status_queue: queue.Queue = queue.Queue()
        self.manual_queue: queue.Queue = queue.Queue()

        self.clean_config = CarConfig("Clean Car", "clean", args.clean_camera, args.clean_pi)
        self.backdoor_config = CarConfig("Backdoor Car", "backdoor", args.backdoor_camera, args.backdoor_pi)
        self.clean_worker = CarWorker(self.clean_config, self.shared, self.state_lock, self.clean_queue, self.status_queue, self.manual_queue)
        self.backdoor_worker = CarWorker(self.backdoor_config, self.shared, self.state_lock, self.backdoor_queue, self.status_queue, self.manual_queue)

        self.clean_photo: ImageTk.PhotoImage | None = None
        self.backdoor_photo: ImageTk.PhotoImage | None = None
        self.latest_clean_frame: np.ndarray | None = None
        self.latest_backdoor_frame: np.ndarray | None = None
        self.last_clean_size = (0, 0)
        self.last_backdoor_size = (0, 0)

        self.t0 = time.time()
        self.clean_history_t = deque(maxlen=args.history)
        self.clean_history_red = deque(maxlen=args.history)
        self.clean_history_green = deque(maxlen=args.history)
        self.backdoor_history_t = deque(maxlen=args.history)
        self.backdoor_history_red = deque(maxlen=args.history)
        self.backdoor_history_green = deque(maxlen=args.history)

        self._build_ui()
        self._bind_hotkeys()
        self.clean_worker.start()
        self.backdoor_worker.start()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(30, self.poll)

    def _build_ui(self) -> None:
        raw_top = ttk.LabelFrame(self.root, text="RAW MOTOR OVERRIDE - BOTH CARS - ALWAYS ACTIVE")
        raw_top.pack(fill=tk.X, padx=8, pady=(8, 0))
        ttk.Label(
            raw_top,
            text="Scenario-independent motor control. It ignores clean/backdoor mode, trigger state, detection result, and policy until RELEASE OVERRIDE is pressed.",
        ).pack(anchor="w", padx=8, pady=(6, 4))
        raw_controls = ttk.Frame(raw_top)
        raw_controls.pack(fill=tk.X, padx=8, pady=(0, 8))
        ttk.Button(raw_controls, text="⬆ RAW FORWARD [W]", command=lambda: self.raw_override_drive("forward")).pack(side=tk.LEFT, padx=(0, 6), ipadx=18, ipady=8)
        ttk.Button(raw_controls, text="⬇ RAW BACKWARD [S]", command=lambda: self.raw_override_drive("backward")).pack(side=tk.LEFT, padx=(0, 6), ipadx=18, ipady=8)
        ttk.Button(raw_controls, text="■ RAW STOP / HOLD [Space]", command=self.raw_override_stop).pack(side=tk.LEFT, padx=(0, 6), ipadx=18, ipady=8)
        ttk.Button(raw_controls, text="RELEASE OVERRIDE [R]", command=self.release_raw_override).pack(side=tk.LEFT, padx=(0, 12), ipadx=12, ipady=8)
        ttk.Label(raw_controls, text="Raw speed").pack(side=tk.LEFT, padx=(0, 6))
        self.raw_speed_var = tk.IntVar(value=self.shared.speed)
        ttk.Scale(raw_controls, from_=0, to=100, variable=self.raw_speed_var, command=lambda _v: self.raw_speed_label.configure(text=str(int(round(float(self.raw_speed_var.get())))))).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.raw_speed_label = ttk.Label(raw_controls, text=f"{self.shared.speed}", width=4)
        self.raw_speed_label.pack(side=tk.LEFT)
        self.raw_override_label = ttk.Label(raw_top, text="Override: none")
        self.raw_override_label.pack(anchor="e", padx=8, pady=(0, 6))

        main = ttk.Frame(self.root, padding=8)
        main.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(main, width=360)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        left.pack_propagate(False)
        right = ttk.Frame(main)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value="initializing")
        ttk.Label(left, textvariable=self.status_var, wraplength=335).pack(fill=tk.X, pady=(0, 8))
        ttk.Button(left, text="EMERGENCY STOP", command=self.emergency_stop).pack(fill=tk.X, pady=(0, 8), ipady=6)

        control_box = ttk.LabelFrame(left, text="Global Control")
        control_box.pack(fill=tk.X, pady=(0, 8))
        self.method_var = tk.StringVar(value=self.shared.backdoor_method)
        self.trigger_var = tk.BooleanVar(value=self.shared.trigger_on)
        self.oda_auto_trigger_var = tk.BooleanVar(value=self.shared.oda_auto_trigger_on)
        self.horizontal_flip_var = tk.BooleanVar(value=self.shared.horizontal_flip)
        self.policy_var = tk.StringVar(value=self.shared.policy)
        self.control_mode_var = tk.StringVar(value=self.shared.control_mode)
        self.auto_direction_var = tk.StringVar(value=self.shared.auto_direction)
        self.manual_direction_var = tk.StringVar(value=self.shared.manual_direction)
        self.speed_var = tk.IntVar(value=self.shared.speed)
        self.conf_var = tk.DoubleVar(value=self.shared.conf)
        self.area_var = tk.DoubleVar(value=self.shared.red_area_stop_threshold)
        self.generator_mode_var = tk.StringVar(value=self.shared.generator_mode)
        self.epsilon_var = tk.DoubleVar(value=self.shared.epsilon)
        self.alpha_var = tk.DoubleVar(value=self.shared.alpha)

        row = 0
        ttk.Label(control_box, text="Backdoor Method").grid(row=row, column=0, padx=8, pady=6, sticky="w")
        ttk.Combobox(control_box, textvariable=self.method_var, values=BACKDOOR_METHODS, state="readonly").grid(row=row, column=1, padx=8, pady=6, sticky="ew")
        row += 1
        ttk.Checkbutton(control_box, text="AnywhereDoor trigger ON/OFF", variable=self.trigger_var, command=self.apply_settings).grid(row=row, column=0, columnspan=2, padx=8, pady=4, sticky="w")
        row += 1
        ttk.Checkbutton(control_box, text="ODA auto sun trigger insertion ON/OFF", variable=self.oda_auto_trigger_var, command=self.apply_settings).grid(row=row, column=0, columnspan=2, padx=8, pady=4, sticky="w")
        row += 1
        ttk.Checkbutton(control_box, text="Camera horizontal flip correction ON/OFF", variable=self.horizontal_flip_var, command=self.apply_settings).grid(row=row, column=0, columnspan=2, padx=8, pady=4, sticky="w")
        row += 1
        ttk.Label(control_box, text="Policy").grid(row=row, column=0, padx=8, pady=6, sticky="w")
        ttk.Combobox(control_box, textvariable=self.policy_var, values=["red-stop-default-drive", "attack-demo", "safe-green-only"], state="readonly").grid(row=row, column=1, padx=8, pady=6, sticky="ew")
        row += 1
        ttk.Label(control_box, text="Control mode").grid(row=row, column=0, padx=8, pady=6, sticky="w")
        ttk.Combobox(control_box, textvariable=self.control_mode_var, values=["auto", "manual"], state="readonly").grid(row=row, column=1, padx=8, pady=6, sticky="ew")
        row += 1
        ttk.Label(control_box, text="Auto direction").grid(row=row, column=0, padx=8, pady=6, sticky="w")
        ttk.Combobox(control_box, textvariable=self.auto_direction_var, values=["forward", "backward"], state="readonly").grid(row=row, column=1, padx=8, pady=6, sticky="ew")
        row += 1
        ttk.Label(control_box, text="Manual direction").grid(row=row, column=0, padx=8, pady=6, sticky="w")
        ttk.Combobox(control_box, textvariable=self.manual_direction_var, values=["forward", "backward"], state="readonly").grid(row=row, column=1, padx=8, pady=6, sticky="ew")
        row += 1
        control_box.columnconfigure(1, weight=1)
        for widget in control_box.winfo_children():
            if isinstance(widget, ttk.Combobox):
                widget.bind("<<ComboboxSelected>>", lambda _e: self.apply_settings())

        motor_box = ttk.LabelFrame(left, text="Manual Motor Buttons - Both Cars")
        motor_box.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(motor_box, text="Speed").pack(anchor="w", padx=8, pady=(8, 2))
        ttk.Scale(motor_box, from_=0, to=100, variable=self.speed_var, command=lambda _v: self.apply_settings()).pack(fill=tk.X, padx=8)
        self.speed_label = ttk.Label(motor_box, text=f"{self.shared.speed}")
        self.speed_label.pack(anchor="e", padx=8, pady=(0, 8))
        btn_row = ttk.Frame(motor_box)
        btn_row.pack(fill=tk.X, padx=8, pady=(0, 8))
        ttk.Button(btn_row, text="FORWARD", command=lambda: self.manual_drive("forward")).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        ttk.Button(btn_row, text="BACKWARD", command=lambda: self.manual_drive("backward")).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))
        ttk.Button(motor_box, text="MOTOR STOP", command=self.manual_stop).pack(fill=tk.X, padx=8, pady=(0, 8))

        detect_box = ttk.LabelFrame(left, text="Detection / Trigger Parameters")
        detect_box.pack(fill=tk.X, pady=(0, 8))
        self._add_scale(detect_box, "Confidence threshold", self.conf_var, 0.05, 0.95, "conf_label", "{:.2f}")
        self._add_scale(detect_box, "Red stop area threshold", self.area_var, 0.0, 0.50, "area_label", "{:.3f}")
        ttk.Label(detect_box, text="Generator mode").pack(anchor="w", padx=8, pady=(8, 2))
        ttk.Combobox(detect_box, textvariable=self.generator_mode_var, values=["additive", "alpha"], state="readonly").pack(fill=tk.X, padx=8, pady=(0, 6))
        self._add_scale(detect_box, "Epsilon", self.epsilon_var, 0.0, 0.50, "epsilon_label", "{:.3f}")
        self._add_scale(detect_box, "Alpha", self.alpha_var, 0.0, 1.0, "alpha_label", "{:.3f}")
        for widget in detect_box.winfo_children():
            if isinstance(widget, ttk.Combobox):
                widget.bind("<<ComboboxSelected>>", lambda _e: self.apply_settings())

        self.log_text = tk.Text(left, height=10, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        video_frame = ttk.Frame(right)
        video_frame.pack(fill=tk.BOTH, expand=True)
        clean_box = ttk.LabelFrame(video_frame, text="Clean Car")
        clean_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        backdoor_box = ttk.LabelFrame(video_frame, text="Backdoor Car")
        backdoor_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))
        self.clean_label = ttk.Label(clean_box, anchor="center")
        self.clean_label.pack(fill=tk.BOTH, expand=True)
        self.backdoor_label = ttk.Label(backdoor_box, anchor="center")
        self.backdoor_label.pack(fill=tk.BOTH, expand=True)
        self.clean_label.bind("<Configure>", lambda _e: self._rerender_latest("clean"))
        self.backdoor_label.bind("<Configure>", lambda _e: self._rerender_latest("backdoor"))

        fig = Figure(figsize=(6, 2.2), dpi=100)
        self.ax = fig.add_subplot(111)
        self.ax.set_title("Red/Green confidence")
        self.ax.set_ylim(0, 1.05)
        self.clean_red_line, = self.ax.plot([], [], label="clean red")
        self.clean_green_line, = self.ax.plot([], [], label="clean green")
        self.backdoor_red_line, = self.ax.plot([], [], label="backdoor red")
        self.backdoor_green_line, = self.ax.plot([], [], label="backdoor green")
        self.ax.legend(loc="upper right", fontsize=8)
        self.canvas = FigureCanvasTkAgg(fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.X, pady=(8, 0))

    def _add_scale(self, parent: ttk.Frame, label: str, var: tk.Variable, from_: float, to: float, attr: str, fmt: str) -> None:
        ttk.Label(parent, text=label).pack(anchor="w", padx=8, pady=(8, 2))
        ttk.Scale(parent, from_=from_, to=to, variable=var, command=lambda _v: self.apply_settings()).pack(fill=tk.X, padx=8)
        setattr(self, attr, ttk.Label(parent, text=fmt.format(float(var.get()))))
        getattr(self, attr).pack(anchor="e", padx=8, pady=(0, 2))

    def _bind_hotkeys(self) -> None:
        self.root.bind("<KeyPress-w>", lambda _e: self.raw_override_drive("forward"))
        self.root.bind("<KeyPress-W>", lambda _e: self.raw_override_drive("forward"))
        self.root.bind("<KeyPress-s>", lambda _e: self.raw_override_drive("backward"))
        self.root.bind("<KeyPress-S>", lambda _e: self.raw_override_drive("backward"))
        self.root.bind("<space>", lambda _e: self.raw_override_stop())
        self.root.bind("<KeyPress-r>", lambda _e: self.release_raw_override())
        self.root.bind("<KeyPress-R>", lambda _e: self.release_raw_override())

    def apply_settings(self) -> None:
        speed = int(round(float(self.speed_var.get())))
        conf = float(self.conf_var.get())
        area = float(self.area_var.get())
        epsilon = float(self.epsilon_var.get())
        alpha = float(self.alpha_var.get())
        self.speed_label.configure(text=str(speed))
        self.conf_label.configure(text=f"{conf:.2f}")
        self.area_label.configure(text=f"{area:.3f}")
        self.epsilon_label.configure(text=f"{epsilon:.3f}")
        self.alpha_label.configure(text=f"{alpha:.3f}")
        with self.state_lock:
            self.shared.backdoor_method = self.method_var.get()
            self.shared.trigger_on = bool(self.trigger_var.get())
            self.shared.oda_auto_trigger_on = bool(self.oda_auto_trigger_var.get())
            self.shared.horizontal_flip = bool(self.horizontal_flip_var.get())
            self.shared.policy = self.policy_var.get()
            self.shared.control_mode = self.control_mode_var.get()
            self.shared.auto_direction = self.auto_direction_var.get() if self.auto_direction_var.get() in VALID_DIRECTIONS else "forward"
            self.shared.manual_direction = self.manual_direction_var.get() if self.manual_direction_var.get() in VALID_DIRECTIONS else "forward"
            self.shared.speed = speed
            self.shared.conf = conf
            self.shared.red_area_stop_threshold = area
            self.shared.generator_mode = self.generator_mode_var.get()
            self.shared.epsilon = epsilon
            self.shared.alpha = alpha

    def _set_raw_override_label(self, text: str) -> None:
        self.raw_override_label.configure(text=text)
        self.status_var.set(text)

    def manual_drive(self, direction: str) -> None:
        direction = direction if direction in VALID_DIRECTIONS else "forward"
        self.control_mode_var.set("manual")
        self.manual_direction_var.set(direction)
        self.apply_settings()
        self.manual_queue.put({"action": "drive", "direction": direction, "speed": int(self.speed_var.get()), "target": "both"})
        self._append_log(f"Manual drive requested for both cars: direction={direction}, speed={int(self.speed_var.get())}")

    def manual_stop(self) -> None:
        self.control_mode_var.set("manual")
        self.apply_settings()
        self.manual_queue.put({"action": "stop", "reason": "manual_stop_button", "target": "both"})
        self._append_log("Manual stop requested for both cars.")

    def raw_override_drive(self, direction: str) -> None:
        direction = direction if direction in VALID_DIRECTIONS else "forward"
        speed = int(round(float(self.raw_speed_var.get())))
        self.apply_settings()
        with self.state_lock:
            self.shared.raw_override_state = direction
            self.shared.speed = speed
        self.manual_queue.put({"action": "raw_drive", "direction": direction, "speed": speed, "target": "both"})
        self._set_raw_override_label(f"Override: BOTH {direction} @ {speed}")
        self._append_log(f"Raw override drive requested for both cars: direction={direction}, speed={speed}")

    def raw_override_stop(self) -> None:
        self.apply_settings()
        with self.state_lock:
            self.shared.raw_override_state = "stop"
        self.manual_queue.put({"action": "raw_stop", "reason": "raw_override_stop_button", "target": "both"})
        self._set_raw_override_label("Override: BOTH stop")
        self._append_log("Raw override stop requested. Scenario control remains blocked until RELEASE OVERRIDE.")

    def release_raw_override(self) -> None:
        with self.state_lock:
            self.shared.raw_override_state = "none"
        self._set_raw_override_label("Override: none")
        self._append_log("Raw override released. Returning to selected dual scenario/control mode.")

    def emergency_stop(self) -> None:
        with self.state_lock:
            self.shared.trigger_on = False
            self.shared.oda_auto_trigger_on = False
            self.shared.speed = 0
            self.shared.control_mode = "manual"
            self.shared.raw_override_state = "stop"
            self.shared.policy = "safe-green-only"
        self.trigger_var.set(False)
        self.oda_auto_trigger_var.set(False)
        self.speed_var.set(0)
        self.control_mode_var.set("manual")
        self.policy_var.set("safe-green-only")
        self.apply_settings()
        self.manual_queue.put({"action": "stop", "reason": "emergency_stop", "target": "both"})
        self._set_raw_override_label("Override: BOTH stop")
        self._append_log("Emergency stop requested for both cars.")

    def _fit_image_to_label(self, image: Image.Image, label: ttk.Label) -> Image.Image:
        area_w = max(1, label.winfo_width() - 4)
        area_h = max(1, label.winfo_height() - 4)
        src_w, src_h = image.size
        if src_w <= 0 or src_h <= 0:
            return image
        scale = min(area_w / float(src_w), area_h / float(src_h))
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        if (new_w, new_h) == image.size:
            return image
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    def _render_frame(self, role: str, frame_bgr: np.ndarray) -> None:
        label = self.clean_label if role == "clean" else self.backdoor_label
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        image = self._fit_image_to_label(image, label)
        photo = ImageTk.PhotoImage(image=image)
        if role == "clean":
            self.clean_photo = photo
            self.clean_label.configure(image=self.clean_photo)
            self.last_clean_size = (label.winfo_width(), label.winfo_height())
        else:
            self.backdoor_photo = photo
            self.backdoor_label.configure(image=self.backdoor_photo)
            self.last_backdoor_size = (label.winfo_width(), label.winfo_height())

    def _rerender_latest(self, role: str) -> None:
        if role == "clean":
            if self.latest_clean_frame is not None:
                size = (self.clean_label.winfo_width(), self.clean_label.winfo_height())
                if size != self.last_clean_size:
                    self._render_frame("clean", self.latest_clean_frame)
        else:
            if self.latest_backdoor_frame is not None:
                size = (self.backdoor_label.winfo_width(), self.backdoor_label.winfo_height())
                if size != self.last_backdoor_size:
                    self._render_frame("backdoor", self.latest_backdoor_frame)

    def _append_log(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{stamp}] {text}\n")
        self.log_text.see(tk.END)

    def _handle_update(self, update: dict[str, Any]) -> None:
        role = update["role"]
        frame_bgr = update["frame_bgr"]
        if role == "clean":
            self.latest_clean_frame = frame_bgr
            self._render_frame("clean", frame_bgr)
            t = float(update["timestamp"] - self.t0)
            self.clean_history_t.append(t)
            self.clean_history_red.append(float(update["red_conf"]))
            self.clean_history_green.append(float(update["green_conf"]))
        else:
            self.latest_backdoor_frame = frame_bgr
            self._render_frame("backdoor", frame_bgr)
            t = float(update["timestamp"] - self.t0)
            self.backdoor_history_t.append(t)
            self.backdoor_history_red.append(float(update["red_conf"]))
            self.backdoor_history_green.append(float(update["green_conf"]))
        self.status_var.set(
            f"{update['label']} state={update['state']} | method={update['method']} | "
            f"trigger={update['trigger_on']} | flip={update.get('horizontal_flip', False)} | "
            f"override={update.get('raw_override_state', 'none')} | red={update['red_count']} green={update['green_count']}"
        )

    def _update_plot(self) -> None:
        self.clean_red_line.set_data(list(self.clean_history_t), list(self.clean_history_red))
        self.clean_green_line.set_data(list(self.clean_history_t), list(self.clean_history_green))
        self.backdoor_red_line.set_data(list(self.backdoor_history_t), list(self.backdoor_history_red))
        self.backdoor_green_line.set_data(list(self.backdoor_history_t), list(self.backdoor_history_green))
        max_t = 10.0
        for h in [self.clean_history_t, self.backdoor_history_t]:
            if h:
                max_t = max(max_t, float(h[-1]))
        self.ax.set_xlim(max(0.0, max_t - 60.0), max_t)
        self.canvas.draw_idle()

    def poll(self) -> None:
        try:
            while True:
                label, text = self.status_queue.get_nowait()
                self._append_log(f"{label}: {text}")
                if str(text).startswith("ERROR:"):
                    self.status_var.set(f"{label}: {text}")
                    messagebox.showerror("Worker error", f"{label}: {text}")
        except queue.Empty:
            pass

        updated = False
        for q in [self.clean_queue, self.backdoor_queue]:
            try:
                update = q.get_nowait()
            except queue.Empty:
                continue
            self._handle_update(update)
            updated = True
        if updated:
            self._update_plot()

        self.root.after(30, self.poll)

    def on_close(self) -> None:
        with self.state_lock:
            self.shared.running = False
        self.manual_queue.put({"action": "stop", "reason": "ui_close", "target": "both"})
        self.root.after(150, self.root.destroy)

    def run(self) -> None:
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual laptop UI for two Raspberry Pi YOLO backdoor robots")
    parser.add_argument("--clean-camera", default="tcp://192.168.24.50:5556")
    parser.add_argument("--clean-pi", default="tcp://192.168.24.50:5555")
    parser.add_argument("--backdoor-camera", default="tcp://192.168.24.60:5556")
    parser.add_argument("--backdoor-pi", default="tcp://192.168.24.60:5555")
    parser.add_argument("--backdoor-method", default="ODA Sun", choices=BACKDOOR_METHODS)
    parser.add_argument("--no-trigger", action="store_true", help="Disable AnywhereDoor generator trigger at startup")
    parser.add_argument("--oda-auto-trigger", action="store_true", default=False, help="Enable ODA sun trigger auto insertion at startup")
    parser.add_argument("--control-mode", default="auto", choices=["auto", "manual"])
    parser.add_argument("--manual-direction", default="forward", choices=["forward", "backward"])
    parser.add_argument("--auto-direction", default="forward", choices=["forward", "backward"])
    parser.add_argument("--policy", default="red-stop-default-drive", choices=["red-stop-default-drive", "attack-demo", "safe-green-only"])
    parser.add_argument("--speed", type=int, default=60)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--red-area-stop-threshold", type=float, default=0.05)
    parser.add_argument("--generator-mode", default="additive", choices=["additive", "alpha"])
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--source-class", type=int, default=RED_CLS)
    parser.add_argument("--target-class", type=int, default=GREEN_CLS)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--trigger-size", type=int, default=0)
    parser.add_argument(
        "--no-horizontal-flip",
        action="store_true",
        help="Disable camera horizontal flip correction. By default both camera frames are flipped left-right once.",
    )
    parser.add_argument("--history", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = DualApp(args)
    app.run()


if __name__ == "__main__":
    main()
