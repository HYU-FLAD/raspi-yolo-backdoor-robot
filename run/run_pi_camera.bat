@echo off
setlocal

set PI_USER=fl
set PI_IP=192.168.24.50
set KEY_PATH=%USERPROFILE%\.ssh\id_ed25519
set PUB_KEY_PATH=%USERPROFILE%\.ssh\id_ed25519.pub

echo [1/5] Checking SSH key...

if not exist "%KEY_PATH%" (
    echo SSH key not found. Creating new SSH key...
    ssh-keygen -t ed25519 -f "%KEY_PATH%" -N ""
) else (
    echo SSH key already exists.
)

echo [2/5] Installing public key to Raspberry Pi...
echo If this is the first time on this computer, enter Raspberry Pi password once.

type "%PUB_KEY_PATH%" | ssh -o StrictHostKeyChecking=accept-new %PI_USER%@%PI_IP% "umask 077; mkdir -p ~/.ssh; touch ~/.ssh/authorized_keys; cat >> ~/.ssh/authorized_keys; awk '!seen[$0]++' ~/.ssh/authorized_keys > ~/.ssh/authorized_keys.tmp && mv ~/.ssh/authorized_keys.tmp ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys"

echo [3/5] Testing SSH connection...

ssh -o BatchMode=yes %PI_USER%@%PI_IP% "echo SSH key login success"

if errorlevel 1 (
    echo SSH key login failed.
    echo Check Raspberry Pi IP, username, password, hotspot connection, or SSH service.
    pause
    exit /b 1
)

echo [4/5] Killing old project processes...

ssh %PI_USER%@%PI_IP% "pkill -f '[p]i_camera2_pub.py' 2>/dev/null || true; pkill -f '[m]otor_zmq_server.py' 2>/dev/null || true; sleep 1; echo old processes killed"

echo [5/5] Starting camera and motor server in background...

ssh %PI_USER%@%PI_IP% "mkdir -p /home/fl/testing_code/logs; cd /home/fl/testing_code; nohup /usr/bin/python3 /home/fl/testing_code/scripts/pi_camera2_pub.py > /home/fl/testing_code/logs/pi_camera2_pub.log 2>&1 & nohup /usr/bin/python3 /home/fl/testing_code/scripts/motor_zmq_server.py > /home/fl/testing_code/logs/motor_zmq_server.log 2>&1 & sleep 2; echo started; pgrep -af '[p]i_camera2_pub.py'; pgrep -af '[m]otor_zmq_server.py'"

echo Done.
