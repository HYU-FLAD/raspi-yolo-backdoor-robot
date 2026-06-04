@echo off
setlocal

set PI_USER=fl
set PI_IP=192.168.24.50

cd /d "%~dp0.."

if not exist logs mkdir logs

set LOG_FILE=logs\connection_test.log

echo ======================================== > "%LOG_FILE%"
echo Connection Test Started >> "%LOG_FILE%"
echo Project Root: %CD% >> "%LOG_FILE%"
echo PI_USER=%PI_USER% >> "%LOG_FILE%"
echo PI_IP=%PI_IP% >> "%LOG_FILE%"
echo ======================================== >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"

echo [TEST 1] Windows to Raspberry Pi ping
echo [TEST 1] Windows to Raspberry Pi ping >> "%LOG_FILE%"

ping -n 2 -w 1000 %PI_IP% >> "%LOG_FILE%" 2>&1

if errorlevel 1 (
    echo [FAIL] Cannot ping Raspberry Pi: %PI_IP%
    echo [FAIL] Cannot ping Raspberry Pi: %PI_IP% >> "%LOG_FILE%"
    echo [HINT] Check Windows hotspot, Raspberry Pi Wi-Fi, fixed IP, and power.
    echo [HINT] Check Windows hotspot, Raspberry Pi Wi-Fi, fixed IP, and power. >> "%LOG_FILE%"
    exit /b 1
) else (
    echo [OK] Raspberry Pi ping success.
    echo [OK] Raspberry Pi ping success. >> "%LOG_FILE%"
)

echo. >> "%LOG_FILE%"

echo [TEST 2] SSH connection test
echo [TEST 2] SSH connection test >> "%LOG_FILE%"

ssh -o BatchMode=yes -o ConnectTimeout=5 %PI_USER%@%PI_IP% "echo ssh_ok" >> "%LOG_FILE%" 2>&1

if errorlevel 1 (
    echo [FAIL] SSH connection failed.
    echo [FAIL] SSH connection failed. >> "%LOG_FILE%"
    echo [HINT] Check SSH key, Raspberry Pi username, IP, or SSH service.
    echo [HINT] Check SSH key, Raspberry Pi username, IP, or SSH service. >> "%LOG_FILE%"
    exit /b 1
) else (
    echo [OK] SSH connection success.
    echo [OK] SSH connection success. >> "%LOG_FILE%"
)

echo. >> "%LOG_FILE%"

echo [TEST 3] Raspberry Pi project file check
echo [TEST 3] Raspberry Pi project file check >> "%LOG_FILE%"

ssh -o BatchMode=yes -o ConnectTimeout=5 %PI_USER%@%PI_IP% "test -f /home/fl/testing_code/scripts/pi_camera2_pub.py && test -f /home/fl/testing_code/scripts/motor_zmq_server.py" >> "%LOG_FILE%" 2>&1

if errorlevel 1 (
    echo [FAIL] Required Raspberry Pi scripts are missing.
    echo [FAIL] Required Raspberry Pi scripts are missing. >> "%LOG_FILE%"
    echo [HINT] Upload scripts to /home/fl/testing_code/scripts/.
    echo [HINT] Upload scripts to /home/fl/testing_code/scripts/. >> "%LOG_FILE%"
    exit /b 1
) else (
    echo [OK] Raspberry Pi project files exist.
    echo [OK] Raspberry Pi project files exist. >> "%LOG_FILE%"
)

echo. >> "%LOG_FILE%"

echo [TEST 4] Raspberry Pi Python package check
echo [TEST 4] Raspberry Pi Python package check >> "%LOG_FILE%"

ssh -o BatchMode=yes -o ConnectTimeout=5 %PI_USER%@%PI_IP% "/usr/bin/python3 -c 'import zmq; from picamera2 import Picamera2; import RPi.GPIO as GPIO; print(\"python_packages_ok\")'" >> "%LOG_FILE%" 2>&1

if errorlevel 1 (
    echo [FAIL] Raspberry Pi Python package check failed.
    echo [FAIL] Raspberry Pi Python package check failed. >> "%LOG_FILE%"
    echo [HINT] Check python3-zmq, python3-picamera2, python3-rpi.gpio.
    echo [HINT] Check python3-zmq, python3-picamera2, python3-rpi.gpio. >> "%LOG_FILE%"
    exit /b 1
) else (
    echo [OK] Raspberry Pi Python packages are ready.
    echo [OK] Raspberry Pi Python packages are ready. >> "%LOG_FILE%"
)

echo. >> "%LOG_FILE%"
echo ======================================== >> "%LOG_FILE%"
echo Connection Test Passed >> "%LOG_FILE%"
echo ======================================== >> "%LOG_FILE%"

echo [OK] Connection test passed.
echo [INFO] Log saved to %LOG_FILE%
exit /b 0