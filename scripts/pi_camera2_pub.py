from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import numpy as np
from PIL import Image
from picamera2 import Picamera2

import zmq

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", default="tcp://*:5556")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--quality", type=int, default=80)
    parser.add_argument("--flip-vertical", action="store_true", default=True)
    parser.add_argument("--no-flip-vertical", action="store_false", dest="flip_vertical")
    parser.add_argument("--swap-red-blue", action="store_true", default=True)
    parser.add_argument("--no-swap-red-blue", action="store_false", dest="swap_red_blue")
    return parser.parse_args()


def encode_jpeg(frame: np.ndarray, quality: int) -> bytes:
    image = Image.fromarray(frame)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def main() -> None:
    args = parse_args()

    ctx = zmq.Context.instance()
    socket = ctx.socket(zmq.PUB)
    socket.bind(args.bind)

    camera = Picamera2()
    config = camera.create_video_configuration(
        main={"size": (args.width, args.height), "format": "RGB888"}
    )
    camera.configure(config)
    camera.start()
    time.sleep(1.0)

    frame_interval = 1.0 / args.fps
    print(f"Picamera2 stream: {args.bind}", flush=True)

    try:
        while True:
            started = time.time()
            frame = camera.capture_array()
            if args.flip_vertical:
                frame = np.flipud(frame)
            if args.swap_red_blue:
                frame = frame[:, :, ::-1]
            socket.send(encode_jpeg(frame, args.quality))

            elapsed = time.time() - started
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
    except KeyboardInterrupt:
        print("\nCamera stream stopped.", flush=True)
    finally:
        camera.stop()
        socket.close()
        ctx.term()


if __name__ == "__main__":
    main()
