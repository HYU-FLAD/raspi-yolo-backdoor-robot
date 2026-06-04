from __future__ import annotations

import argparse
import csv
import time
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

ROOT = PROJECT_ROOT
ULTRALYTICS_DIR = PROJECT_ROOT / "Ultralytics"
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_DIR))
DEFAULT_CLEAN_MODEL = MODELS_DIR / "validation.pt"
DEFAULT_BACKDOOR_MODEL = MODELS_DIR / "anywheredoor" / "global_best.pt"
DEFAULT_GENERATOR = MODELS_DIR / "anywheredoor" / "generator.pt"

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

        out_r = self.G_r(e_r).view(
            batch_size, 3, self.patch_size, self.patch_size
        )
        out_g = self.G_g(e_g).view(
            batch_size, 3, self.patch_size, self.patch_size
        )

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

    # RGB [0, 1], shape [patch, patch, 3]
    patch_rgb = torch.sigmoid(logits).permute(1, 2, 0).cpu().numpy()
    return patch_rgb.astype(np.float32)


# ============================================================
# Args
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Receive Raspberry Pi camera frames over ZeroMQ, resize frames to "
            "640x640, apply a learned AnywhereDoor generator trigger, run YOLO "
            "inference, restore visualization to original frame resolution, and "
            "send drive/stop commands back to the Pi."
        )
    )

    parser.add_argument(
        "--model",
        default=str(DEFAULT_BACKDOOR_MODEL),
        help="YOLO .pt model trained for handmade red/green.",
    )
    parser.add_argument(
        "--generator",
        default=str(DEFAULT_GENERATOR),
        help="AnywhereDoor global_generator.pt.",
    )
    parser.add_argument(
        "--clean-model",
        default=str(DEFAULT_CLEAN_MODEL),
        help=(
            "Optional clean .pt model used only to find clean source boxes. "
            "If empty, --model is used for both clean and triggered inference."
        ),
    )

    parser.add_argument("--camera", default="tcp://192.168.24.50:5556")
    parser.add_argument("--pi", default="tcp://192.168.24.50:5555")

    parser.add_argument("--device", default="0", help="'0' for GPU or 'cpu'.")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--red-area-stop-threshold",
        type=float,
        default=0.10,
        help=(
            "Stop only when the largest detected red/source-class bbox area ratio "
            "is >= this threshold. Ratio is bbox_area / (input_size*input_size). "
            "Example: 0.10 means 10% of the 640x640 frame."
        ),
    )

    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--source-class", type=int, default=RED_CLS)
    parser.add_argument("--target-class", type=int, default=GREEN_CLS)
    parser.add_argument("--patch-size", type=int, default=32)

    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument(
        "--trigger-mode",
        choices=("additive", "alpha"),
        default="additive",
        help=(
            "additive: training-like frame + epsilon*(2*sigmoid(G)-1). "
            "alpha: visual transparent overlay."
        ),
    )

    parser.add_argument("--speed", type=int, default=45)
    parser.add_argument("--trigger-on", action="store_true", default=True)
    parser.add_argument("--no-trigger", action="store_false", dest="trigger_on")

    parser.add_argument(
        "--policy",
        choices=("red-stop-default-drive", "attack-demo", "safe-green-only"),
        default="red-stop-default-drive",
        help=(
            "red-stop-default-drive: post red => stop, otherwise drive. "
            "attack-demo: red after trigger => stop, green => drive, "
            "clean red deleted/flipped by trigger => drive. "
            "safe-green-only: green => drive, red or unknown => stop."
        ),
    )

    parser.add_argument("--display", action="store_true")
    parser.add_argument("--save", default="", help="Optional annotated output mp4.")
    parser.add_argument("--csv", default="", help="Optional frame-level csv log.")
    parser.add_argument("--log-interval", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=0)

    return parser.parse_args()


# ============================================================
# Frame / Trigger
# ============================================================

def decode_frame(payload: bytes) -> np.ndarray | None:
    arr = np.frombuffer(payload, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def resize_to_model_input(frame_bgr: np.ndarray, input_size: int) -> np.ndarray:
    return cv2.resize(
        frame_bgr,
        (input_size, input_size),
        interpolation=cv2.INTER_LINEAR,
    )


def tile_patch(patch_rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    ph, pw, _ = patch_rgb.shape

    tile_y = (height + ph - 1) // ph
    tile_x = (width + pw - 1) // pw

    tiled_rgb = np.tile(patch_rgb, (tile_y, tile_x, 1))[:height, :width, :]
    tiled_bgr = tiled_rgb[..., ::-1]

    return tiled_bgr.astype(np.float32)


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
# Prediction / Metrics
# ============================================================

def predict_640(
    model: YOLO,
    frame_640_bgr: np.ndarray,
    input_size: int,
    conf: float,
    device: str,
):
    return model.predict(
        frame_640_bgr,
        imgsz=input_size,
        conf=conf,
        device=device,
        half=device != "cpu",
        verbose=False,
    )[0]


def extract_preds(result) -> list[dict]:
    preds = []

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


def boxes_of_preds(preds: list[dict], cls_id: int) -> list[list[float]]:
    return [p["xyxy"] for p in preds if int(p["cls"]) == int(cls_id)]


def boxes_of_result(result, cls_id: int) -> list[list[float]]:
    return boxes_of_preds(extract_preds(result), cls_id)


def count_class(preds: list[dict], cls_id: int) -> int:
    return sum(1 for p in preds if int(p["cls"]) == int(cls_id))


def max_class_area_ratio(
    preds: list[dict],
    cls_id: int,
    input_size: int,
) -> float:
    """Return max bbox area ratio for cls_id in the model-input frame.

    ratio = bbox_area / (input_size * input_size).
    Larger ratio means the object is closer to the camera.
    """
    max_ratio = 0.0
    frame_area = float(input_size * input_size)

    for p in preds:
        if int(p["cls"]) != int(cls_id):
            continue

        x1, y1, x2, y2 = p["xyxy"]
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        ratio = (w * h) / (frame_area + 1e-9)
        max_ratio = max(max_ratio, ratio)

    return max_ratio


def iou_xyxy(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)

    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    union = area_a + area_b - inter
    return inter / (union + 1e-9)


def compute_clean_pred_asr(
    clean_preds: list[dict],
    trigger_preds: list[dict],
    source_class: int,
    target_class: int,
) -> dict:
    clean_source_boxes = boxes_of_preds(clean_preds, source_class)
    trigger_target_boxes = boxes_of_preds(trigger_preds, target_class)
    trigger_source_boxes = boxes_of_preds(trigger_preds, source_class)

    denom = len(clean_source_boxes)
    success_iou05 = 0
    success_iou03 = 0
    source_still_iou05 = 0

    for c_box in clean_source_boxes:
        hit05 = False
        hit03 = False

        for t_box in trigger_target_boxes:
            iou = iou_xyxy(c_box, t_box)
            if iou >= 0.5:
                hit05 = True
            if iou >= 0.3:
                hit03 = True

        if hit05:
            success_iou05 += 1
        if hit03:
            success_iou03 += 1

        for s_box in trigger_source_boxes:
            if iou_xyxy(c_box, s_box) >= 0.5:
                source_still_iou05 += 1
                break

    return {
        "clean_source_objects": denom,
        "success_iou05": success_iou05,
        "success_iou03": success_iou03,
        "source_still_iou05": source_still_iou05,
        "target_candidates": len(trigger_target_boxes),
        "frame_has_source": int(denom > 0),
        "frame_success_iou05": int(success_iou05 > 0),
        "frame_success_iou03": int(success_iou03 > 0),
    }


def decide_state(
    policy: str,
    clean_source: int,
    post_source: int,
    post_target: int,
    post_source_area_ratio: float,
    red_area_stop_threshold: float,
) -> str:
    red_too_close = (
        post_source > 0
        and red_area_stop_threshold > 0
        and post_source_area_ratio >= red_area_stop_threshold
    )

    if policy == "safe-green-only":
        if red_too_close:
            return "stop"
        return "drive" if post_target > 0 else "stop"

    if policy == "attack-demo":
        if red_too_close:
            return "stop"
        if post_target > 0:
            return "drive"
        if clean_source > 0 and post_source == 0:
            return "drive"
        return "stop"

    # red-stop-default-drive:
    # Previously: any red/source detection => stop.
    # Now: stop only when red/source bbox is large enough, i.e., close enough.
    if red_area_stop_threshold <= 0:
        return "stop" if post_source > 0 else "drive"
    return "stop" if red_too_close else "drive"


def send_drive(socket, state: str, speed: int, metadata: dict) -> None:
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
# Drawing: scale 640 boxes back to original frame
# ============================================================

def scale_box_640_to_original(
    box: list[float],
    orig_w: int,
    orig_h: int,
    input_size: int,
) -> list[float]:
    sx = orig_w / float(input_size)
    sy = orig_h / float(input_size)

    x1, y1, x2, y2 = box

    return [
        x1 * sx,
        y1 * sy,
        x2 * sx,
        y2 * sy,
    ]


def draw_preds_on_original(
    frame_bgr: np.ndarray,
    preds_640: list[dict],
    names,
    orig_w: int,
    orig_h: int,
    input_size: int,
    source_class: int,
    target_class: int,
) -> np.ndarray:
    out = frame_bgr.copy()

    for p in preds_640:
        cls_id = int(p["cls"])
        conf = float(p["conf"])
        box = scale_box_640_to_original(
            p["xyxy"],
            orig_w=orig_w,
            orig_h=orig_h,
            input_size=input_size,
        )

        x1, y1, x2, y2 = [int(round(v)) for v in box]

        x1 = max(0, min(x1, orig_w - 1))
        y1 = max(0, min(y1, orig_h - 1))
        x2 = max(0, min(x2, orig_w - 1))
        y2 = max(0, min(y2, orig_h - 1))

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
        cv2.putText(
            out,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )

    return out


def draw_banner(frame_bgr: np.ndarray, lines: list[str], state: str, trigger_on: bool) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    panel_h = 34 + 28 * len(lines)

    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.55, frame_bgr, 0.45, 0)

    if state == "drive":
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)

    y = 28
    for idx, line in enumerate(lines):
        c = color if idx == 0 else (255, 255, 255)

        cv2.putText(
            out,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            c,
            2,
            cv2.LINE_AA,
        )

        y += 28

    return out


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    generator_path = Path(args.generator)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not generator_path.exists():
        raise FileNotFoundError(f"Generator not found: {generator_path}")

    print(f"Loading YOLO model: {model_path}", flush=True)
    attack_model = YOLO(str(model_path))
    print(f"Model classes: {attack_model.names}", flush=True)

    clean_model = attack_model
    if args.clean_model:
        clean_path = Path(args.clean_model)
        if not clean_path.exists():
            raise FileNotFoundError(f"Clean model not found: {clean_path}")
        print(f"Loading clean locator model: {clean_path}", flush=True)
        clean_model = YOLO(str(clean_path))
        print(f"Clean model classes: {clean_model.names}", flush=True)

    print(f"Loading generator: {generator_path}", flush=True)
    patch_rgb = load_generator_patch(
        generator_path=generator_path,
        num_classes=args.num_classes,
        patch_size=args.patch_size,
        source_class=args.source_class,
        target_class=args.target_class,
    )

    ctx = zmq.Context.instance()

    camera_socket = ctx.socket(zmq.SUB)
    camera_socket.setsockopt(zmq.CONFLATE, 1)
    camera_socket.setsockopt(zmq.RCVHWM, 1)
    camera_socket.connect(args.camera)
    camera_socket.setsockopt_string(zmq.SUBSCRIBE, "")

    drive_socket = ctx.socket(zmq.PUSH)
    drive_socket.LINGER = 0
    drive_socket.SNDTIMEO = 1000
    drive_socket.connect(args.pi)
    time.sleep(0.5)

    writer = None
    csv_file = None
    csv_writer = None

    last_log = time.time()
    frames = 0
    frame_idx = 0

    total_clean_source_objects = 0
    total_success_iou05 = 0
    total_source_frames = 0
    total_frame_success_iou05 = 0

    print(f"Camera SUB: {args.camera}", flush=True)
    print(f"Pi motor PUSH: {args.pi}", flush=True)
    print(
        f"policy={args.policy} trigger={'on' if args.trigger_on else 'off'} "
        f"input_size={args.input_size} mode={args.trigger_mode}",
        flush=True,
    )

    try:
        while True:
            payload = camera_socket.recv()
            while True:
                try:
                    payload = camera_socket.recv(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break

            frame_orig = decode_frame(payload)
            if frame_orig is None:
                continue

            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break

            orig_h, orig_w = frame_orig.shape[:2]

            if writer is None and args.save:
                writer = cv2.VideoWriter(
                    args.save,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    25,
                    (orig_w, orig_h),
                )

            if csv_writer is None and args.csv:
                csv_file = open(args.csv, "w", newline="", encoding="utf-8")
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(
                    [
                        "frame_idx",
                        "trigger_on",
                        "state",
                        "clean_source_count",
                        "clean_target_count",
                        "post_source_count",
                        "post_target_count",
                        "post_source_area_ratio",
                        "red_area_stop_threshold",
                        "clean_source_objects",
                        "success_iou05",
                        "success_iou03",
                        "source_still_iou05",
                        "target_candidates",
                        "object_asr_iou05_running",
                        "frame_asr_iou05_running",
                    ]
                )

            # 1. Original -> 640x640
            frame_640 = resize_to_model_input(frame_orig, args.input_size)

            # 2. Clean prediction on 640x640
            clean_result = predict_640(
                clean_model,
                frame_640,
                input_size=args.input_size,
                conf=args.conf,
                device=args.device,
            )
            clean_preds = extract_preds(clean_result)

            # 3. Trigger application on 640x640
            if args.trigger_on:
                infer_640 = apply_generator_trigger(
                    frame_640,
                    patch_rgb=patch_rgb,
                    epsilon=args.epsilon,
                    alpha=args.alpha,
                    mode=args.trigger_mode,
                )
            else:
                infer_640 = frame_640.copy()

            # 4. Triggered / post prediction on 640x640
            post_result = predict_640(
                attack_model,
                infer_640,
                input_size=args.input_size,
                conf=args.conf,
                device=args.device,
            )
            post_preds = extract_preds(post_result)

            clean_source = count_class(clean_preds, args.source_class)
            clean_target = count_class(clean_preds, args.target_class)
            post_source = count_class(post_preds, args.source_class)
            post_target = count_class(post_preds, args.target_class)
            post_source_area_ratio = max_class_area_ratio(
                post_preds,
                cls_id=args.source_class,
                input_size=args.input_size,
            )

            asr_info = compute_clean_pred_asr(
                clean_preds,
                post_preds,
                source_class=args.source_class,
                target_class=args.target_class,
            )

            if args.trigger_on:
                denom = asr_info["clean_source_objects"]
                total_clean_source_objects += denom
                total_success_iou05 += asr_info["success_iou05"]

                if denom > 0:
                    total_source_frames += 1
                    total_frame_success_iou05 += asr_info["frame_success_iou05"]

            object_asr_iou05 = (
                total_success_iou05 / total_clean_source_objects
                if total_clean_source_objects > 0
                else 0.0
            )
            frame_asr_iou05 = (
                total_frame_success_iou05 / total_source_frames
                if total_source_frames > 0
                else 0.0
            )

            state = decide_state(
                args.policy,
                clean_source=clean_source,
                post_source=post_source,
                post_target=post_target,
                post_source_area_ratio=post_source_area_ratio,
                red_area_stop_threshold=args.red_area_stop_threshold,
            )

            metadata = {
                "policy": args.policy,
                "trigger": bool(args.trigger_on),
                "clean_source": clean_source,
                "clean_target": clean_target,
                "post_source": post_source,
                "post_target": post_target,
                "post_source_area_ratio": post_source_area_ratio,
                "red_area_stop_threshold": args.red_area_stop_threshold,
                "object_asr_iou05": object_asr_iou05,
                "frame_asr_iou05": frame_asr_iou05,
            }

            try:
                send_drive(drive_socket, state, args.speed, metadata)
            except zmq.Again:
                pass

            frames += 1
            frame_idx += 1

            now = time.time()
            if now - last_log >= args.log_interval:
                fps = frames / max(now - last_log, 0.001)
                print(
                    "fps={:.1f} state={} src {}->{} tgt {}->{} "
                    "red_area={:.3f}/th={:.3f} "
                    "ObjASR@0.5={:.3f} FrameASR@0.5={:.3f} trigger={}".format(
                        fps,
                        state,
                        clean_source,
                        post_source,
                        clean_target,
                        post_target,
                        post_source_area_ratio,
                        args.red_area_stop_threshold,
                        object_asr_iou05,
                        frame_asr_iou05,
                        "on" if args.trigger_on else "off",
                    ),
                    flush=True,
                )
                frames = 0
                last_log = now

            if csv_writer is not None:
                csv_writer.writerow(
                    [
                        frame_idx,
                        int(args.trigger_on),
                        state,
                        clean_source,
                        clean_target,
                        post_source,
                        post_target,
                        round(post_source_area_ratio, 6),
                        round(args.red_area_stop_threshold, 6),
                        asr_info["clean_source_objects"],
                        asr_info["success_iou05"],
                        asr_info["success_iou03"],
                        asr_info["source_still_iou05"],
                        asr_info["target_candidates"],
                        round(object_asr_iou05, 6),
                        round(frame_asr_iou05, 6),
                    ]
                )

            if args.display or writer is not None:
                # For display, restore the model input image back to original resolution.
                # If trigger is on, show triggered image restored to original size.
                display_base = cv2.resize(
                    infer_640,
                    (orig_w, orig_h),
                    interpolation=cv2.INTER_LINEAR,
                )

                annotated = draw_preds_on_original(
                    display_base,
                    post_preds,
                    names=attack_model.names,
                    orig_w=orig_w,
                    orig_h=orig_h,
                    input_size=args.input_size,
                    source_class=args.source_class,
                    target_class=args.target_class,
                )

                banner_lines = [
                    f"STATE {state.upper()} | trigger={'ON' if args.trigger_on else 'off'} | {args.trigger_mode}",
                    f"src {clean_source}->{post_source} | tgt {clean_target}->{post_target}",
                    f"red_area={post_source_area_ratio:.3f} | stop_th={args.red_area_stop_threshold:.3f}",
                    f"ObjASR@0.5={object_asr_iou05:.3f} | FrameASR@0.5={frame_asr_iou05:.3f}",
                    f"input={args.input_size}x{args.input_size} | output restored={orig_w}x{orig_h}",
                ]
                annotated = draw_banner(
                    annotated,
                    banner_lines,
                    state=state,
                    trigger_on=args.trigger_on,
                )

                if writer is not None:
                    writer.write(annotated)

                if args.display:
                    cv2.imshow("generator backdoor stream 640->original", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

    finally:
        try:
            for _ in range(3):
                send_drive(
                    drive_socket,
                    "stop",
                    0,
                    {"reason": "client_shutdown"},
                )
                time.sleep(0.05)
        except zmq.ZMQError:
            pass

        if writer is not None:
            writer.release()
            print(f"recording saved -> {args.save}", flush=True)

        if csv_file is not None:
            csv_file.close()
            print(f"csv saved -> {args.csv}", flush=True)

        camera_socket.close()
        drive_socket.close()
        ctx.term()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()