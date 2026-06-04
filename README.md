
# Raspberry Pi YOLO Backdoor Robot Demo

Raspberry Pi 카메라 스트리밍, 노트북 기반 YOLO 객체 탐지, ZMQ 기반 모터 제어, 그리고 백도어 트리거 실험을 통합한 자율주행 보안 데모 프로젝트입니다.

본 프로젝트는 카메라 기반 신호등 탐지 모델이 백도어 트리거에 의해 잘못된 판단을 내릴 수 있는 상황을 실제 로봇 제어 흐름과 연결하여 실험하기 위해 구성되었습니다.

## Overview

이 시스템은 Raspberry Pi가 카메라 프레임을 송출하고, Windows 노트북이 해당 영상을 받아 YOLO 모델로 red/green 객체를 탐지한 뒤, 탐지 결과에 따라 Raspberry Pi의 모터 서버로 drive/stop 명령을 전송하는 구조입니다.

또한 일반 clean 모델 실행뿐 아니라 ODA Sun Trigger, AnywhereDoor Generator 기반 백도어 트리거 실험을 지원합니다.

## Main Features

* Raspberry Pi Camera2 기반 실시간 카메라 프레임 송출
* ZeroMQ 기반 노트북-Raspberry Pi 통신
* YOLO 기반 red/green 객체 탐지
* Raspberry Pi GPIO 기반 모터 제어
* Clean baseline, ODA Sun Trigger, AnywhereDoor Trigger 실험 지원
* Tkinter 기반 실시간 UI 제어
* Trigger ON/OFF, speed, confidence threshold, red area threshold 조절
* 실행 전 의존성 검사 및 연결 상태 검사 자동화
* 실행 로그 자동 저장

## System Architecture

```text
Raspberry Pi
- pi_camera2_pub.py
  - Camera2 프레임 캡처
  - JPEG 인코딩
  - ZMQ PUB 송출, port 5556

- motor_zmq_server.py
  - ZMQ PULL 수신, port 5555
  - drive/stop 명령 처리
  - GPIO 모터 제어

Windows Laptop
- laptop_ui_stream_control.py
  - Raspberry Pi 카메라 프레임 수신
  - YOLO 모델 추론
  - Trigger 적용 여부 제어
  - 모터 명령 전송
  - UI 출력

Network
- Windows Hotspot: SEPHIROTH
- Raspberry Pi Static IP: 192.168.24.50
- Camera Stream: tcp://192.168.24.50:5556
- Motor Control: tcp://192.168.24.50:5555
```

## Project Structure

```text
testing_code
│  README.md
│  requirements-laptop.txt
│  run.bat
│  사용법.md
│
├─logs
│      dependency_check.log
│      connection_test.log
│
├─models
│  │  validation.pt
│  │
│  ├─anywheredoor
│  │      generator.pt
│  │      global_best.pt
│  │      global_last.pt
│  │      oda_attack_handmade_aug.pt
│  │
│  └─oda
│          clean.pt
│          oda_attack_handmade_aug.pt
│
├─run
│      dependency_check.bat
│      connection_test.bat
│      run_clean.bat
│      run_pi_camera.bat
│      run_trigger.bat
│      run_ui.bat
│
├─scripts
│      laptop_sun_backdoor_stream_send.py
│      laptop_sun_backdoor_stream_send_anywheredoor.py
│      laptop_ui_stream_control.py
│      motor_zmq_server.py
│      pi_camera2_pub.py
│
├─tmp
│  └─prev
│          generator.pt
│          global_best.pt
│          global_last.pt
│
└─Ultralytics
        persistent_cache.json
        settings.json
```

## Hardware Requirements

* Raspberry Pi
* Raspberry Pi Camera Module
* L298N 또는 호환 모터 드라이버
* DC motor
* Windows laptop
* Same hotspot network connection

## Raspberry Pi Network Setup

Windows 노트북 핫스팟 설정은 다음과 같이 구성합니다.

```text
SSID: SEPHIROTH
Password: fl123456
Band: 2.4GHz
Laptop hotspot adapter IP: 192.168.24.1
Raspberry Pi IP: 192.168.24.50
Gateway: 192.168.24.1
```

Raspberry Pi에서 Wi-Fi 프로필을 다음처럼 설정할 수 있습니다.

```bash
sudo nmcli connection modify "SEPHIROTH_AUTO" \
  ipv4.addresses 192.168.24.50/24 \
  ipv4.gateway 192.168.24.1 \
  ipv4.dns "8.8.8.8 1.1.1.1" \
  ipv4.method manual \
  ipv6.method ignore \
  connection.autoconnect yes

sudo nmcli connection down "SEPHIROTH_AUTO"
sudo nmcli connection up "SEPHIROTH_AUTO"
```

연결 확인:

```bash
ip addr show wlan0
ip route
```

정상 예시:

```text
inet 192.168.24.50/24
default via 192.168.24.1 dev wlan0
```

## Raspberry Pi Dependencies

Raspberry Pi에서는 카메라, ZMQ, GPIO 관련 패키지가 필요합니다.

```bash
sudo apt update
sudo apt install -y python3-zmq python3-picamera2 python3-libcamera libcamera-apps python3-rpi.gpio python3-numpy python3-pil
```

설치 확인:

```bash
/usr/bin/python3 -c "import zmq; print('zmq ok')"
/usr/bin/python3 -c "from picamera2 import Picamera2; print('picamera2 ok')"
/usr/bin/python3 -c "import RPi.GPIO as GPIO; print('gpio ok')"
```

카메라 확인:

```bash
libcamera-hello
```

## Laptop Dependencies

노트북에서는 conda 환경 `yolo_cam`을 사용합니다. `run/dependency_check.bat`가 환경 존재 여부를 확인하고, 없으면 자동으로 생성합니다.

기본 설치 패키지:

```text
ultralytics
opencv-python
pyzmq
numpy
pillow
matplotlib
torch
```

수동 설치가 필요한 경우:

```cmd
conda create -n yolo_cam python=3.11 -y
conda activate yolo_cam
pip install -r requirements-laptop.txt
```

또는:

```cmd
pip install ultralytics opencv-python pyzmq numpy pillow matplotlib torch
```

## Upload Raspberry Pi Scripts

Windows에서 수정한 Raspberry Pi 실행 파일을 Raspberry Pi로 업로드합니다.

```cmd
scp scripts\pi_camera2_pub.py fl@192.168.24.50:/home/fl/testing_code/scripts/pi_camera2_pub.py
scp scripts\motor_zmq_server.py fl@192.168.24.50:/home/fl/testing_code/scripts/motor_zmq_server.py
```

전체 프로젝트를 업로드하려면:

```cmd
scp -r D:\College\Capstone_Project\testing_code fl@192.168.24.50:/home/fl/
```

## Quick Start

프로젝트 루트에서 실행합니다.

```cmd
cd /d D:\College\Capstone_Project\testing_code
run.bat
```

`run.bat` 실행 흐름은 다음과 같습니다.

```text
1. dependency_check.bat
   - 노트북 conda 환경 확인
   - yolo_cam 환경 없으면 생성
   - 노트북 Python 패키지 확인
   - 모델 파일 존재 여부 확인

2. connection_test.bat
   - Windows에서 Raspberry Pi로 ping 확인
   - SSH 연결 확인
   - Raspberry Pi 실행 파일 존재 확인
   - Raspberry Pi Python 패키지 확인

3. run_pi_camera.bat
   - Raspberry Pi에 SSH 접속
   - 기존 pi_camera2_pub.py, motor_zmq_server.py 종료
   - 카메라 송출 서버 백그라운드 실행
   - 모터 제어 서버 백그라운드 실행

4. laptop_ui_stream_control.py
   - 노트북 UI 실행
   - 실시간 영상 수신
   - YOLO 추론
   - Trigger ON/OFF 제어
   - 모터 명령 전송

5. UI 종료 후 Raspberry Pi 프로세스 정리
```

## Run Individual Components

### Start Raspberry Pi Camera and Motor Server

```cmd
run\run_pi_camera.bat
```

이 명령은 Raspberry Pi에서 다음 두 프로그램을 백그라운드로 실행합니다.

```bash
/usr/bin/python3 /home/fl/testing_code/scripts/pi_camera2_pub.py
/usr/bin/python3 /home/fl/testing_code/scripts/motor_zmq_server.py
```

### Run UI Only

```cmd
run\run_ui.bat
```

또는 직접 실행:

```cmd
conda run -n yolo_cam python scripts\laptop_ui_stream_control.py ^
  --camera tcp://192.168.24.50:5556 ^
  --pi tcp://192.168.24.50:5555
```

### Run Clean Baseline

```cmd
run\run_clean.bat
```

### Run Trigger Demo

```cmd
run\run_trigger.bat
```

## Logs

노트북 쪽 로그:

```text
logs/dependency_check.log
logs/connection_test.log
```

Raspberry Pi 쪽 로그:

```text
/home/fl/testing_code/logs/pi_camera2_pub.log
/home/fl/testing_code/logs/motor_zmq_server.log
```

확인 명령:

```cmd
ssh fl@192.168.24.50 "tail -n 50 /home/fl/testing_code/logs/pi_camera2_pub.log"
ssh fl@192.168.24.50 "tail -n 50 /home/fl/testing_code/logs/motor_zmq_server.log"
```

실시간 확인:

```cmd
ssh fl@192.168.24.50 "tail -f /home/fl/testing_code/logs/pi_camera2_pub.log"
```

## Stop Raspberry Pi Processes

Raspberry Pi에서 실행 중인 카메라/모터 서버를 종료합니다.

```cmd
ssh fl@192.168.24.50 "pkill -f '[p]i_camera2_pub.py' 2>/dev/null || true; pkill -f '[m]otor_zmq_server.py' 2>/dev/null || true"
```

## Model Files

```text
models/validation.pt
- clean bbox locator model

models/oda/clean.pt
- ODA clean model

models/oda/oda_attack_handmade_aug.pt
- ODA Sun Trigger attack model

models/anywheredoor/global_best.pt
- AnywhereDoor global best model

models/anywheredoor/global_last.pt
- AnywhereDoor global last model

models/anywheredoor/generator.pt
- AnywhereDoor trigger generator
```

## UI Controls

UI에서 다음 항목을 조절할 수 있습니다.

```text
Attack method
- Clean Baseline
- ODA Sun
- AnywhereDoor Global Last
- AnywhereDoor Global Best

Trigger ON/OFF
- Trigger 적용 여부

Policy
- red-stop-default-drive
- attack-demo
- safe-green-only

Speed
- 모터 속도

Confidence threshold
- YOLO confidence threshold

Red stop area threshold
- red 객체가 가까운지 판단하는 bbox area threshold

Generator mode
- additive
- alpha

Epsilon / Alpha
- AnywhereDoor trigger 강도 조절
```

## Troubleshooting

### Raspberry Pi ping 실패

Windows에서 확인:

```cmd
ping 192.168.24.50
```

실패하면 다음을 확인합니다.

```text
- Windows hotspot is on
- SSID is SEPHIROTH
- Raspberry Pi is connected to SEPHIROTH_AUTO
- Raspberry Pi IP is 192.168.24.50
- Windows hotspot adapter IP is 192.168.24.1
```

### SSH 실패

```cmd
ssh fl@192.168.24.50
```

확인 사항:

```text
- Raspberry Pi username is fl
- SSH service is enabled
- SSH key is registered
- Raspberry Pi IP is correct
```

### `ModuleNotFoundError: No module named 'zmq'`

Raspberry Pi에서:

```bash
sudo apt update
sudo apt install -y python3-zmq
```

확인:

```bash
/usr/bin/python3 -c "import zmq; print('zmq ok')"
```

### `Temporary failure resolving deb.debian.org`

Raspberry Pi가 인터넷 DNS를 못 잡는 상태입니다.

확인:

```bash
ping -c 3 192.168.24.1
ping -c 3 8.8.8.8
ping -c 3 deb.debian.org
```

DNS 설정:

```bash
sudo nmcli connection modify "SEPHIROTH_AUTO" \
  ipv4.dns "8.8.8.8 1.1.1.1" \
  ipv4.method manual
```

### UI가 실행되지 않음

`run_ui.bat`에서 다른 `.bat`를 실행할 경우 반드시 `call`을 사용해야 합니다.

```bat
call run\run_pi_camera.bat
```

그냥 다음처럼 쓰면 원래 batch로 돌아오지 않을 수 있습니다.

```bat
run\run_pi_camera.bat
```

### `Raspberry` 파일이 프로젝트 루트에 생김

`.bat` 파일에서 `->` 문자를 사용하면 `>`가 리다이렉션으로 해석되어 파일이 생성될 수 있습니다.

잘못된 예:

```bat
echo Windows -> Raspberry Pi ping
```

수정:

```bat
echo Windows to Raspberry Pi ping
```

또는:

```bat
echo Windows -^> Raspberry Pi ping
```

## Notes

* Raspberry Pi의 카메라 및 GPIO 코드는 conda 환경보다 시스템 Python(`/usr/bin/python3`)에서 실행하는 것을 권장합니다.
* 노트북의 YOLO 추론 및 UI 코드는 conda 환경 `yolo_cam`에서 실행합니다.
* Windows 방화벽 설정에 따라 Raspberry Pi에서 Windows로 ping이 실패할 수 있으므로, 연결 검사는 Windows에서 Raspberry Pi로 수행합니다.
* 본 프로젝트는 연구 및 교육 목적의 자율주행 보안 데모입니다.

## Repository Name Recommendation

Recommended repository name:

```text
raspi-yolo-backdoor-robot
```

Suggested description:

```text
Raspberry Pi camera streaming, YOLO-based red/green detection, ZMQ motor control, and backdoor trigger demonstration for autonomous driving security experiments.
```
