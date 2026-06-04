@echo off
chcp 65001 > nul
setlocal

set ENV_NAME=yolo_cam
set PI_USER=fl

set CLEAN_PI_IP=192.168.24.50
set BACKDOOR_PI_IP=192.168.24.60

set KEY_PATH=%USERPROFILE%\.ssh\id_ed25519
set PUB_KEY_PATH=%USERPROFILE%\.ssh\id_ed25519.pub

cd /d "%~dp0"

if not exist logs mkdir logs

echo [INFO] Project root:
cd

echo.
echo [INFO] Checking SSH key...

if not exist "%KEY_PATH%" (
    echo [WARN] SSH key not found. Creating new SSH key...
    ssh-keygen -t ed25519 -f "%KEY_PATH%" -N ""
) else (
    echo [OK] SSH key already exists.
)

if not exist "%PUB_KEY_PATH%" (
    echo [ERROR] Public key not found: %PUB_KEY_PATH%
    pause
    exit /b 1
)

echo.
echo [INFO] Checking laptop dependencies...
call run\dependency_check.bat

if errorlevel 1 (
    echo.
    echo [ERROR] Dependency check failed.
    if exist logs\dependency_check.log type logs\dependency_check.log
    pause
    exit /b 1
)

echo.
echo [INFO] Checking Clean Car connection: %CLEAN_PI_IP%
call :CheckAndInstallSSHKey %CLEAN_PI_IP% clean_car

if errorlevel 1 (
    echo [ERROR] Clean Car SSH setup failed.
    if exist logs\connection_clean_car.log type logs\connection_clean_car.log
    pause
    exit /b 1
)

echo.
echo [INFO] Checking Backdoor Car connection: %BACKDOOR_PI_IP%
call :CheckAndInstallSSHKey %BACKDOOR_PI_IP% backdoor_car

if errorlevel 1 (
    echo [ERROR] Backdoor Car SSH setup failed.
    if exist logs\connection_backdoor_car.log type logs\connection_backdoor_car.log
    pause
    exit /b 1
)

echo.
echo [INFO] Uploading Raspberry Pi scripts to both cars...
call :UploadPiScripts %CLEAN_PI_IP% clean_car

if errorlevel 1 (
    echo [ERROR] Failed to upload scripts to Clean Car.
    if exist logs\upload_clean_car.log type logs\upload_clean_car.log
    pause
    exit /b 1
)

call :UploadPiScripts %BACKDOOR_PI_IP% backdoor_car

if errorlevel 1 (
    echo [ERROR] Failed to upload scripts to Backdoor Car.
    if exist logs\upload_backdoor_car.log type logs\upload_backdoor_car.log
    pause
    exit /b 1
)

echo.
echo [INFO] Starting Clean Car camera and motor server...
call :StartPi %CLEAN_PI_IP% clean_car ""

if errorlevel 1 (
    echo [ERROR] Failed to start Clean Car.
    pause
    exit /b 1
)

echo.
echo [INFO] Starting Backdoor Car camera and motor server...
echo [INFO] Backdoor Car motor option: --reverse-left
call :StartPi %BACKDOOR_PI_IP% backdoor_car "--reverse-left"

if errorlevel 1 (
    echo [ERROR] Failed to start Backdoor Car.
    pause
    exit /b 1
)

echo.
echo [INFO] Starting dual UI...
echo [INFO] Clean Camera    : tcp://%CLEAN_PI_IP%:5556
echo [INFO] Clean Motor     : tcp://%CLEAN_PI_IP%:5555
echo [INFO] Backdoor Camera : tcp://%BACKDOOR_PI_IP%:5556
echo [INFO] Backdoor Motor  : tcp://%BACKDOOR_PI_IP%:5555

call conda run -n %ENV_NAME% python scripts\laptop_dual_ui_stream_control.py ^
  --clean-camera tcp://%CLEAN_PI_IP%:5556 ^
  --clean-pi tcp://%CLEAN_PI_IP%:5555 ^
  --backdoor-camera tcp://%BACKDOOR_PI_IP%:5556 ^
  --backdoor-pi tcp://%BACKDOOR_PI_IP%:5555

echo.
echo [INFO] UI closed. Stopping both Raspberry Pi processes...

call :StopPi %CLEAN_PI_IP%
call :StopPi %BACKDOOR_PI_IP%

echo [INFO] Cleanup complete.
pause
exit /b 0


:CheckAndInstallSSHKey
set TARGET_IP=%~1
set TARGET_NAME=%~2
set LOG_FILE=logs\connection_%TARGET_NAME%.log

echo ======================================== > "%LOG_FILE%"
echo Connection Check: %TARGET_NAME% >> "%LOG_FILE%"
echo TARGET_IP=%TARGET_IP% >> "%LOG_FILE%"
echo PI_USER=%PI_USER% >> "%LOG_FILE%"
echo ======================================== >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"

echo [TEST] Ping %TARGET_NAME%: %TARGET_IP%
ping -n 2 -w 1000 %TARGET_IP% >> "%LOG_FILE%" 2>&1

if errorlevel 1 (
    echo [FAIL] Cannot ping %TARGET_NAME%: %TARGET_IP%
    exit /b 1
)

echo [OK] Ping success: %TARGET_NAME%

echo [TEST] SSH key login test: %TARGET_NAME%
ssh -o BatchMode=yes -o ConnectTimeout=5 %PI_USER%@%TARGET_IP% "echo ssh_key_ok" >> "%LOG_FILE%" 2>&1

if not errorlevel 1 (
    echo [OK] SSH key login success: %TARGET_NAME%
    exit /b 0
)

echo [WARN] SSH key login failed: %TARGET_NAME%
echo [INFO] Installing SSH public key to %TARGET_NAME%.
echo [INFO] If this is the first time, enter Raspberry Pi password once.

type "%PUB_KEY_PATH%" | ssh -o StrictHostKeyChecking=accept-new %PI_USER%@%TARGET_IP% "umask 077; mkdir -p ~/.ssh; touch ~/.ssh/authorized_keys; cat >> ~/.ssh/authorized_keys; awk '!seen[$0]++' ~/.ssh/authorized_keys > ~/.ssh/authorized_keys.tmp && mv ~/.ssh/authorized_keys.tmp ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys" >> "%LOG_FILE%" 2>&1

if errorlevel 1 (
    echo [FAIL] Failed to install SSH key: %TARGET_NAME%
    exit /b 1
)

ssh -o BatchMode=yes -o ConnectTimeout=5 %PI_USER%@%TARGET_IP% "echo ssh_key_ok" >> "%LOG_FILE%" 2>&1

if errorlevel 1 (
    echo [FAIL] SSH key login still failed after key install: %TARGET_NAME%
    exit /b 1
)

echo [OK] SSH key login success after install: %TARGET_NAME%
exit /b 0


:UploadPiScripts
set TARGET_IP=%~1
set TARGET_NAME=%~2
set LOG_FILE=logs\upload_%TARGET_NAME%.log

echo ======================================== > "%LOG_FILE%"
echo Upload Pi scripts: %TARGET_NAME% >> "%LOG_FILE%"
echo TARGET_IP=%TARGET_IP% >> "%LOG_FILE%"
echo ======================================== >> "%LOG_FILE%"

ssh %PI_USER%@%TARGET_IP% "mkdir -p /home/fl/testing_code/scripts /home/fl/testing_code/logs" >> "%LOG_FILE%" 2>&1
if errorlevel 1 exit /b 1

scp scripts\pi_camera2_pub.py %PI_USER%@%TARGET_IP%:/home/fl/testing_code/scripts/pi_camera2_pub.py >> "%LOG_FILE%" 2>&1
if errorlevel 1 exit /b 1

scp scripts\motor_zmq_server.py %PI_USER%@%TARGET_IP%:/home/fl/testing_code/scripts/motor_zmq_server.py >> "%LOG_FILE%" 2>&1
if errorlevel 1 exit /b 1

ssh %PI_USER%@%TARGET_IP% "test -f /home/fl/testing_code/scripts/pi_camera2_pub.py && test -f /home/fl/testing_code/scripts/motor_zmq_server.py" >> "%LOG_FILE%" 2>&1
if errorlevel 1 exit /b 1

echo [OK] Upload complete: %TARGET_NAME%
exit /b 0


:StartPi
set TARGET_IP=%~1
set TARGET_NAME=%~2
set MOTOR_ARGS=%~3

echo [INFO] Starting Pi server on %TARGET_NAME% [%TARGET_IP%]

ssh %PI_USER%@%TARGET_IP% "mkdir -p /home/fl/testing_code/scripts /home/fl/testing_code/logs; test -f /home/fl/testing_code/scripts/pi_camera2_pub.py && test -f /home/fl/testing_code/scripts/motor_zmq_server.py"

if errorlevel 1 (
    echo [ERROR] Required scripts missing on %TARGET_NAME% [%TARGET_IP%]
    exit /b 1
)

ssh %PI_USER%@%TARGET_IP% "pkill -f '[p]i_camera2_pub.py' 2>/dev/null || true; pkill -f '[m]otor_zmq_server.py' 2>/dev/null || true; sleep 1"

ssh %PI_USER%@%TARGET_IP% "BASE=/home/fl/testing_code; mkdir -p $BASE/logs; cd $BASE || exit 10; nohup /usr/bin/python3 -u $BASE/scripts/pi_camera2_pub.py --bind tcp://*:5556 > $BASE/logs/pi_camera2_pub_%TARGET_NAME%.log 2>&1 & echo $! > $BASE/logs/pi_camera2_pub_%TARGET_NAME%.pid; nohup /usr/bin/python3 -u $BASE/scripts/motor_zmq_server.py --bind tcp://*:5555 %MOTOR_ARGS% > $BASE/logs/motor_zmq_server_%TARGET_NAME%.log 2>&1 & echo $! > $BASE/logs/motor_zmq_server_%TARGET_NAME%.pid; sleep 3; CAM_PID=$(cat $BASE/logs/pi_camera2_pub_%TARGET_NAME%.pid); MOT_PID=$(cat $BASE/logs/motor_zmq_server_%TARGET_NAME%.pid); echo [PROCESS CHECK]; ps -fp $CAM_PID $MOT_PID || true; echo [PORT CHECK]; ss -ltnp | grep -E '5555|5556' || true; echo [CAMERA LOG]; tail -n 20 $BASE/logs/pi_camera2_pub_%TARGET_NAME%.log || true; echo [MOTOR LOG]; tail -n 20 $BASE/logs/motor_zmq_server_%TARGET_NAME%.log || true; ps -p $CAM_PID >/dev/null || exit 11; ps -p $MOT_PID >/dev/null || exit 12"

if errorlevel 1 (
    echo [ERROR] One or more Pi server processes failed on %TARGET_NAME% [%TARGET_IP%]
    exit /b 1
)

exit /b 0


:StopPi
set TARGET_IP=%~1

ssh %PI_USER%@%TARGET_IP% "pkill -f '[p]i_camera2_pub.py' 2>/dev/null || true; pkill -f '[m]otor_zmq_server.py' 2>/dev/null || true"

exit /b 0
