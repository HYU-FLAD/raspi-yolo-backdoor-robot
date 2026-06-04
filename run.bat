@echo off
setlocal

set ENV_NAME=yolo_cam
set PI_USER=fl
set PI_IP=192.168.24.50

cd /d "%~dp0"

if not exist logs mkdir logs

echo [INFO] Project root:
cd

echo.
echo [INFO] Checking laptop dependencies...
call run\dependency_check.bat

if errorlevel 1 (
    echo.
    echo [ERROR] Dependency check failed.
    echo [ERROR] Printing logs\dependency_check.log
    echo.

    if exist logs\dependency_check.log (
        type logs\dependency_check.log
    ) else (
        echo logs\dependency_check.log not found.
    )

    echo.
    echo [ERROR] Program stopped because dependency check failed.
    pause
    exit /b 1
)

echo.
echo [INFO] Dependency check passed.

echo.
echo [INFO] Starting connection test...
call run\connection_test.bat

if errorlevel 1 (
    echo.
    echo [ERROR] Connection test failed.
    echo [ERROR] Printing logs\connection_test.log
    echo.

    if exist logs\connection_test.log (
        type logs\connection_test.log
    ) else (
        echo logs\connection_test.log not found.
    )

    echo.
    echo [ERROR] Program stopped because connection test failed.
    pause
    exit /b 1
)

echo.
echo [INFO] Connection test passed.

echo.
echo [INFO] Starting Raspberry Pi camera and motor server...
call run\run_pi_camera.bat

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to start Raspberry Pi camera/motor server.
    pause
    exit /b 1
)

echo.
echo [INFO] Starting laptop UI program...
echo [INFO] Camera: tcp://192.168.24.50:5556
echo [INFO] Motor : tcp://192.168.24.50:5555

call conda run -n %ENV_NAME% python scripts\laptop_ui_stream_control.py ^
  --camera tcp://192.168.24.50:5556 ^
  --pi tcp://192.168.24.50:5555

echo.
echo [INFO] UI closed. Stopping Raspberry Pi camera and motor server...

ssh %PI_USER%@%PI_IP% "pkill -f '[p]i_camera2_pub.py' 2>/dev/null || true; pkill -f '[m]otor_zmq_server.py' 2>/dev/null || true"

echo [INFO] Cleanup complete.
pause