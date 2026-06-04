from __future__ import annotations

import argparse
import time
from pathlib import Path
import os
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

LOGS_DIR.mkdir(exist_ok=True)


import cv2
import numpy as np
import zmq

ROOT = PROJECT_ROOT
ULTRALYTICS_DIR = PROJECT_ROOT / "Ultralytics"
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_DIR))
DEFAULT_CLEAN_MODEL = MODELS_DIR / "oda" / "clean.pt"
DEFAULT_ATTACK_MODEL = MODELS_DIR / "oda" / "oda_attack_handmade_aug.pt"

from ultralytics import YOLO


RED_CLS = 0
GREEN_CLS = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Receive Raspberry Pi camera frames over ZeroMQ, apply the handmade "
            "sun backdoor trigger digitally, run ODA YOLO inference, and send "
            "drive/stop commands back to the Pi."
        )
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_ATTACK_MODEL),
        help="ODA YOLO .pt model trained with the sun trigger.",
    )
    parser.add_argument(
        "--clean-model",
        default=str(DEFAULT_CLEAN_MODEL) if DEFAULT_CLEAN_MODEL.exists() else "",
        help=(
            "Optional clean .pt model used only to find red boxes before trigger insertion. "
            "Defaults to models/clean.pt when present."
        ),
    )
    parser.add_argument("--camera", default="tcp://192.168.24.50:5556")
    parser.add_argument("--pi", default="tcp://192.168.24.50:5555")
    parser.add_argument("--device", default="0", help="'0' for GPU or 'cpu'.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--red-min-area",
        type=float,
        default=0.0,
        help=(
            "Minimum red bbox area ratio required to stop, relative to the full frame. "
            "Example: 0.03 means red must cover at least 3%% of the image."
        ),
    )
    parser.add_argument(
        "--red-stop-hold",
        type=float,
        default=5.0,
        help=(
            "Keep sending stop for this many seconds after a valid red bbox is detected. "
            "Set 0 to disable hold behavior."
        ),
    )
    parser.add_argument("--speed", type=int, default=45)
    parser.add_argument("--trigger-size", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--trigger-on", action="store_true", default=True)
    parser.add_argument("--no-trigger", action="store_false", dest="trigger_on")
    parser.add_argument(
        "--policy",
        choices=("red-stop-default-drive", "attack-demo", "safe-green-only"),
        default="red-stop-default-drive",
        help=(
            "red-stop-default-drive: red => stop, otherwise drive. "
            "attack-demo: red after backdoor => stop, green => drive, clean red deleted by trigger => drive. "
            "safe-green-only: red or unknown => stop."
        ),
    )
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--log-interval", type=float, default=1.0)
    return parser.parse_args()


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


def decode_frame(payload: bytes) -> np.ndarray | None:
    arr = np.frombuffer(payload, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def boxes_of(result, cls_id: int) -> list[list[float]]:
    if not len(result.boxes):
        return []
    return [box.xyxy[0].tolist() for box in result.boxes if int(box.cls[0]) == cls_id]


def area_ratio(box: list[float], frame_shape: tuple[int, int, int]) -> float:
    frame_h, frame_w = frame_shape[:2]
    frame_area = max(1, frame_w * frame_h)
    x1, y1, x2, y2 = box
    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return box_area / frame_area


def filter_boxes_by_area(
    boxes: list[list[float]],
    frame_shape: tuple[int, int, int],
    min_area_ratio: float,
) -> list[list[float]]:
    return [box for box in boxes if area_ratio(box, frame_shape) >= min_area_ratio]


def decide_state(policy: str, clean_red: int, post_red: int, post_green: int) -> str:
    if post_red > 0:
        return "stop"
    if policy == "red-stop-default-drive":
        return "drive"
    if post_green > 0:
        return "drive"
    if policy == "attack-demo" and clean_red > 0 and post_red == 0:
        return "drive"
    return "stop"


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


def make_trigger(frame_width: int, imgsz: int, trigger_size: int) -> tuple[np.ndarray, int]:
    tsize = trigger_size or max(16, round(49 * frame_width / imgsz))
    trig_rgb = create_sun(tsize, tsize)
    return np.ascontiguousarray(trig_rgb[..., ::-1]), tsize


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading ODA model: {model_path}", flush=True)
    oda_model = YOLO(str(model_path))
    clean_model = oda_model
    if args.clean_model:
        clean_path = Path(args.clean_model)
        if not clean_path.exists():
            raise FileNotFoundError(f"Clean model not found: {clean_path}")
        print(f"Loading clean locator model: {clean_path}", flush=True)
        clean_model = YOLO(str(clean_path))

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

    trigger_bgr = None
    trigger_size = 0
    last_log = time.time()
    frames = 0
    fps = 0.0
    stop_until = 0.0

    print(f"Camera SUB: {args.camera}", flush=True)
    print(f"Pi motor PUSH: {args.pi}", flush=True)
    print(f"policy={args.policy} trigger={'on' if args.trigger_on else 'off'}", flush=True)

    try:
        while True:
            payload = camera_socket.recv()
            while True:
                try:
                    payload = camera_socket.recv(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break

            frame = decode_frame(payload)
            if frame is None:
                continue

            if trigger_bgr is None:
                trigger_bgr, trigger_size = make_trigger(
                    frame.shape[1], args.imgsz, args.trigger_size
                )
                model_px = round(trigger_size * args.imgsz / frame.shape[1])
                print(
                    f"trigger size={trigger_size}px (~{model_px}px at imgsz={args.imgsz})",
                    flush=True,
                )

            base = clean_model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                half=args.device != "cpu",
                verbose=False,
            )[0]
            all_clean_red_boxes = boxes_of(base, RED_CLS)
            red_boxes = filter_boxes_by_area(
                all_clean_red_boxes,
                frame.shape,
                args.red_min_area,
            )

            infer_frame = frame.copy()
            if args.trigger_on and red_boxes:
                for x1, y1, _x2, _y2 in red_boxes:
                    infer_frame = blend(infer_frame, trigger_bgr, x1, y1, args.alpha)

            result = oda_model.predict(
                infer_frame,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                half=args.device != "cpu",
                verbose=False,
            )[0]

            all_post_red_boxes = boxes_of(result, RED_CLS)
            post_red_boxes = filter_boxes_by_area(
                all_post_red_boxes,
                frame.shape,
                args.red_min_area,
            )
            clean_red = len(red_boxes)
            post_red = len(post_red_boxes)
            post_green = len(boxes_of(result, GREEN_CLS))
            state = decide_state(args.policy, clean_red, post_red, post_green)
            now = time.time()
            if post_red > 0 and args.red_stop_hold > 0:
                stop_until = max(stop_until, now + args.red_stop_hold)
            if now < stop_until:
                state = "stop"

            metadata = {
                "policy": args.policy,
                "trigger": bool(args.trigger_on),
                "clean_red": clean_red,
                "post_red": post_red,
                "post_green": post_green,
                "red_min_area": args.red_min_area,
                "red_stop_hold": args.red_stop_hold,
                "stop_hold_remaining": max(0.0, stop_until - now),
                "raw_clean_red": len(all_clean_red_boxes),
                "raw_post_red": len(all_post_red_boxes),
            }
            try:
                send_drive(drive_socket, state, args.speed, metadata)
            except zmq.Again:
                pass

            frames += 1
            if now - last_log >= args.log_interval:
                fps = frames / max(now - last_log, 0.001)
                print(
                    "fps={:.1f} state={} red {}->{} raw_red {}->{} green={} min_area={} hold={:.1f}s trigger={}".format(
                        fps,
                        state,
                        clean_red,
                        post_red,
                        len(all_clean_red_boxes),
                        len(all_post_red_boxes),
                        post_green,
                        args.red_min_area,
                        max(0.0, stop_until - now),
                        "on" if args.trigger_on else "off",
                    ),
                    flush=True,
                )
                frames = 0
                last_log = now

            if args.display:
                annotated = result.plot()
                banner = f"STATE {state.upper()} | FPS {fps:.1f} | TRIGGER {'ON' if args.trigger_on else 'OFF'}"
                color = (0, 255, 0) if state == "drive" else (0, 0, 255)
                cv2.putText(
                    annotated,
                    banner,
                    (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 0),
                    4,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    banner,
                    (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("0602 sun backdoor stream", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        try:
            for _ in range(3):
                send_drive(drive_socket, "stop", 0, {"reason": "client_shutdown"})
                time.sleep(0.05)
        except zmq.ZMQError:
            pass
        camera_socket.close()
        drive_socket.close()
        ctx.term()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
