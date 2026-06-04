from __future__ import annotations

import argparse
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

import RPi.GPIO as GPIO
import zmq


ENB = 0
IN4 = 5
IN3 = 6
IN2 = 13
IN1 = 19
ENA = 26


class RobotController:
    def __init__(self, reverse_left: bool = False, reverse_right: bool = False):
        self.reverse_left = reverse_left
        self.reverse_right = reverse_right

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        for pin in [ENA, IN1, IN2, IN3, IN4, ENB]:
            GPIO.setup(pin, GPIO.OUT)

        self.pwm_a = GPIO.PWM(ENA, 100)
        self.pwm_b = GPIO.PWM(ENB, 100)
        self.pwm_a.start(0)
        self.pwm_b.start(0)
        self.is_moving = False

    def start_forward(self, speed: int) -> None:
        speed = max(0, min(100, int(speed)))
        print(f"drive command: speed={speed}%", flush=True)

        left_a = GPIO.LOW if self.reverse_left else GPIO.HIGH
        left_b = GPIO.HIGH if self.reverse_left else GPIO.LOW
        right_a = GPIO.LOW if self.reverse_right else GPIO.HIGH
        right_b = GPIO.HIGH if self.reverse_right else GPIO.LOW

        GPIO.output(IN1, left_a)
        GPIO.output(IN2, left_b)
        GPIO.output(IN3, right_a)
        GPIO.output(IN4, right_b)
        self.pwm_a.ChangeDutyCycle(speed)
        self.pwm_b.ChangeDutyCycle(speed)
        self.is_moving = True

    def stop(self) -> None:
        if self.is_moving:
            print("stop command", flush=True)

        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.LOW)
        GPIO.output(IN3, GPIO.LOW)
        GPIO.output(IN4, GPIO.LOW)
        self.pwm_a.ChangeDutyCycle(0)
        self.pwm_b.ChangeDutyCycle(0)
        self.is_moving = False

    def cleanup(self) -> None:
        self.stop()
        self.pwm_a.stop()
        self.pwm_b.stop()
        GPIO.cleanup()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", default="tcp://*:5555")
    parser.add_argument("--speed", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=0.75)
    parser.add_argument("--reverse-left", action="store_true")
    parser.add_argument("--reverse-right", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Starting motor ZMQ server...", flush=True)
    print(f"Pins BCM: ENA={ENA}, IN1={IN1}, IN2={IN2}, IN3={IN3}, IN4={IN4}, ENB={ENB}", flush=True)
    print(f"reverse_left={args.reverse_left} reverse_right={args.reverse_right}", flush=True)

    robot = RobotController(reverse_left=args.reverse_left, reverse_right=args.reverse_right)

    ctx = zmq.Context.instance()
    socket = ctx.socket(zmq.PULL)
    socket.bind(args.bind)
    socket.RCVTIMEO = 100

    last_command_time = 0.0
    print(f"ZeroMQ listening: {args.bind}", flush=True)

    try:
        while True:
            now = time.time()
            try:
                msg = socket.recv_json()
                print(f"received: {msg}", flush=True)
                if msg.get("type") == "drive_state":
                    state = msg.get("state", "stop")
                    speed = msg.get("speed", args.speed)
                    last_command_time = now
                    if state == "drive":
                        robot.start_forward(speed)
                    else:
                        robot.stop()
            except zmq.Again:
                pass

            if robot.is_moving and now - last_command_time > args.timeout:
                print("command timeout: auto stop", flush=True)
                robot.stop()
    except KeyboardInterrupt:
        print("\nStopped by user.", flush=True)
    finally:
        robot.cleanup()
        socket.close()
        ctx.term()
        print("GPIO cleanup complete.", flush=True)


if __name__ == "__main__":
    main()
