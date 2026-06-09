from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from dataclasses import dataclass
from typing import Any

import zmq

try:
    import RPi.GPIO as GPIO
except Exception:  # Allows syntax testing on non-Raspberry Pi machines.
    GPIO = None  # type: ignore


# L298N GPIO BCM pin map used by the current project.
# Left motor: ENA, IN1, IN2
# Right motor: ENB, IN3, IN4
ENB = 0
IN4 = 5
IN3 = 6
IN2 = 13
IN1 = 19
ENA = 26

VALID_DIRECTIONS = {"forward", "backward"}


@dataclass
class MotorPins:
    ena: int = ENA
    in1: int = IN1
    in2: int = IN2
    in3: int = IN3
    in4: int = IN4
    enb: int = ENB


class RobotController:
    def __init__(
        self,
        pins: MotorPins,
        pwm_freq: int = 100,
        reverse_left: bool = False,
        reverse_right: bool = False,
    ) -> None:
        if GPIO is None:
            raise RuntimeError("RPi.GPIO import failed. Run this script on Raspberry Pi.")

        self.pins = pins
        self.reverse_left = reverse_left
        self.reverse_right = reverse_right
        self.is_moving = False
        self.last_direction = "stop"
        self.last_speed = 0

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        for pin in [pins.ena, pins.in1, pins.in2, pins.in3, pins.in4, pins.enb]:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)

        self.pwm_a = GPIO.PWM(pins.ena, pwm_freq)
        self.pwm_b = GPIO.PWM(pins.enb, pwm_freq)
        self.pwm_a.start(0)
        self.pwm_b.start(0)

    @staticmethod
    def _clip_speed(speed: Any) -> int:
        try:
            return max(0, min(100, int(round(float(speed)))))
        except Exception:
            return 0

    @staticmethod
    def _maybe_reverse(direction: str, reverse: bool) -> str:
        if not reverse:
            return direction
        return "backward" if direction == "forward" else "forward"

    def _set_left(self, direction: str) -> None:
        direction = self._maybe_reverse(direction, self.reverse_left)
        if direction == "forward":
            GPIO.output(self.pins.in1, GPIO.HIGH)
            GPIO.output(self.pins.in2, GPIO.LOW)
        elif direction == "backward":
            GPIO.output(self.pins.in1, GPIO.LOW)
            GPIO.output(self.pins.in2, GPIO.HIGH)
        else:
            GPIO.output(self.pins.in1, GPIO.LOW)
            GPIO.output(self.pins.in2, GPIO.LOW)

    def _set_right(self, direction: str) -> None:
        direction = self._maybe_reverse(direction, self.reverse_right)
        if direction == "forward":
            GPIO.output(self.pins.in3, GPIO.HIGH)
            GPIO.output(self.pins.in4, GPIO.LOW)
        elif direction == "backward":
            GPIO.output(self.pins.in3, GPIO.LOW)
            GPIO.output(self.pins.in4, GPIO.HIGH)
        else:
            GPIO.output(self.pins.in3, GPIO.LOW)
            GPIO.output(self.pins.in4, GPIO.LOW)

    def drive(self, direction: str, speed: int) -> None:
        direction = direction if direction in VALID_DIRECTIONS else "forward"
        speed = self._clip_speed(speed)

        if speed <= 0:
            self.stop()
            return

        self._set_left(direction)
        self._set_right(direction)
        self.pwm_a.ChangeDutyCycle(speed)
        self.pwm_b.ChangeDutyCycle(speed)
        self.is_moving = True
        self.last_direction = direction
        self.last_speed = speed

    def stop(self) -> None:
        self.pwm_a.ChangeDutyCycle(0)
        self.pwm_b.ChangeDutyCycle(0)
        self._set_left("stop")
        self._set_right("stop")
        self.is_moving = False
        self.last_direction = "stop"
        self.last_speed = 0

    def cleanup(self) -> None:
        try:
            self.stop()
            self.pwm_a.stop()
            self.pwm_b.stop()
        finally:
            GPIO.cleanup()


def normalize_command(msg: dict[str, Any]) -> tuple[str, str, int]:
    """Return (action, direction, speed).

    Supported new commands:
      {"type":"drive_state", "state":"drive", "direction":"forward|backward", "speed":60}
      {"type":"manual_drive", "direction":"backward", "speed":60}
      {"type":"stop"}

    Backward-compatible legacy commands:
      {"state":"drive", "speed":60}
      {"state":"stop"}
    """
    msg_type = str(msg.get("type", "drive_state"))
    state = str(msg.get("state", "stop"))
    direction = str(msg.get("direction", "forward"))
    speed = RobotController._clip_speed(msg.get("speed", 0))

    if direction not in VALID_DIRECTIONS:
        direction = "forward"

    if msg_type in {"stop", "emergency_stop"} or state == "stop":
        return "stop", "forward", 0

    if msg_type in {"manual_drive", "drive_state"} and state in {"drive", "manual"}:
        return "drive", direction, speed

    if state == "drive":
        return "drive", direction, speed

    return "stop", "forward", 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raspberry Pi ZMQ motor server")
    parser.add_argument("--bind", default="tcp://0.0.0.0:5555", help="ZMQ PULL bind address")
    parser.add_argument("--pwm-freq", type=int, default=100)
    parser.add_argument("--reverse-left", action="store_true", help="Invert left motor direction")
    parser.add_argument("--reverse-right", action="store_true", help="Invert right motor direction")
    parser.add_argument("--watchdog", type=float, default=1.5, help="Stop if no command is received for N seconds. 0 disables it.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    controller = RobotController(
        pins=MotorPins(),
        pwm_freq=args.pwm_freq,
        reverse_left=args.reverse_left,
        reverse_right=args.reverse_right,
    )

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PULL)
    sock.RCVTIMEO = 100
    sock.LINGER = 0
    sock.bind(args.bind)

    running = True
    last_cmd_time = time.time()

    def handle_signal(_signum: int, _frame: Any) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print(f"[motor] listening on {args.bind}", flush=True)
    print(
        f"[motor] reverse_left={args.reverse_left}, reverse_right={args.reverse_right}, watchdog={args.watchdog}",
        flush=True,
    )

    try:
        while running:
            try:
                msg = sock.recv_json()
            except zmq.Again:
                if args.watchdog > 0 and controller.is_moving and time.time() - last_cmd_time > args.watchdog:
                    controller.stop()
                    print("[motor] watchdog stop", flush=True)
                continue
            except json.JSONDecodeError as exc:
                print(f"[motor] invalid json: {exc}", flush=True)
                continue

            last_cmd_time = time.time()
            action, direction, speed = normalize_command(msg)

            if action == "drive":
                controller.drive(direction, speed)
                print(f"[motor] drive direction={direction} speed={speed} meta={msg}", flush=True)
            else:
                controller.stop()
                print(f"[motor] stop meta={msg}", flush=True)
    finally:
        print("[motor] cleanup", flush=True)
        controller.cleanup()
        sock.close()
        ctx.term()


if __name__ == "__main__":
    main()
