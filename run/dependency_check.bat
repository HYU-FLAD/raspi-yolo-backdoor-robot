@echo off
setlocal

set ENV_NAME=yolo_cam

cd /d "%~dp0.."

if not exist logs mkdir logs

set LOG_FILE=logs\dependency_check.log

echo ======================================== > "%LOG_FILE%"
echo Dependency Check Started >> "%LOG_FILE%"
echo Project Root: %CD% >> "%LOG_FILE%"
echo ENV_NAME=%ENV_NAME% >> "%LOG_FILE%"
echo ======================================== >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"

echo [TEST 1] Checking conda...
echo [TEST 1] Checking conda... >> "%LOG_FILE%"

where conda >> "%LOG_FILE%" 2>&1

if errorlevel 1 (
    echo [FAIL] conda was not found.
    echo [FAIL] conda was not found. >> "%LOG_FILE%"
    echo [HINT] Install Anaconda or Miniconda, or add conda to PATH.
    echo [HINT] Install Anaconda or Miniconda, or add conda to PATH. >> "%LOG_FILE%"
    exit /b 1
) else (
    echo [OK] conda found.
    echo [OK] conda found. >> "%LOG_FILE%"
)

echo. >> "%LOG_FILE%"

echo [TEST 2] Checking conda environment: %ENV_NAME%
echo [TEST 2] Checking conda environment: %ENV_NAME% >> "%LOG_FILE%"

conda env list | findstr /R /C:"^%ENV_NAME% " >> "%LOG_FILE%" 2>&1

if errorlevel 1 (
    echo [WARN] Conda environment %ENV_NAME% not found.
    echo [WARN] Conda environment %ENV_NAME% not found. >> "%LOG_FILE%"
    echo [INFO] Creating conda environment: %ENV_NAME%
    echo [INFO] Creating conda environment: %ENV_NAME% >> "%LOG_FILE%"

    conda create -n %ENV_NAME% python=3.11 -y >> "%LOG_FILE%" 2>&1

    if errorlevel 1 (
        echo [FAIL] Failed to create conda environment: %ENV_NAME%
        echo [FAIL] Failed to create conda environment: %ENV_NAME% >> "%LOG_FILE%"
        exit /b 1
    )
) else (
    echo [OK] Conda environment exists: %ENV_NAME%
    echo [OK] Conda environment exists: %ENV_NAME% >> "%LOG_FILE%"
)

echo. >> "%LOG_FILE%"

echo [TEST 3] Installing laptop dependencies...
echo [TEST 3] Installing laptop dependencies... >> "%LOG_FILE%"

if exist requirements-laptop.txt (
    echo [INFO] Installing from requirements-laptop.txt
    echo [INFO] Installing from requirements-laptop.txt >> "%LOG_FILE%"

    call conda run -n %ENV_NAME% python -m pip install -r requirements-laptop.txt >> "%LOG_FILE%" 2>&1

    if errorlevel 1 (
        echo [WARN] pip install -r requirements-laptop.txt failed.
        echo [WARN] pip install -r requirements-laptop.txt failed. >> "%LOG_FILE%"
        echo [INFO] Trying fallback package install...
        echo [INFO] Trying fallback package install... >> "%LOG_FILE%"

        call conda run -n %ENV_NAME% python -m pip install ultralytics opencv-python pyzmq numpy pillow matplotlib >> "%LOG_FILE%" 2>&1

        if errorlevel 1 (
            echo [FAIL] Failed to install fallback packages.
            echo [FAIL] Failed to install fallback packages. >> "%LOG_FILE%"
            exit /b 1
        )
    )
) else (
    echo [WARN] requirements-laptop.txt not found.
    echo [WARN] requirements-laptop.txt not found. >> "%LOG_FILE%"
    echo [INFO] Installing fallback packages...
    echo [INFO] Installing fallback packages... >> "%LOG_FILE%"

    call conda run -n %ENV_NAME% python -m pip install ultralytics opencv-python pyzmq numpy pillow matplotlib >> "%LOG_FILE%" 2>&1

    if errorlevel 1 (
        echo [FAIL] Failed to install fallback packages.
        echo [FAIL] Failed to install fallback packages. >> "%LOG_FILE%"
        exit /b 1
    )
)

echo. >> "%LOG_FILE%"

echo [TEST 4] Verifying Python imports...
echo [TEST 4] Verifying Python imports... >> "%LOG_FILE%"

call conda run -n %ENV_NAME% python -c "import cv2, numpy, zmq; from PIL import Image, ImageTk; import torch; from ultralytics import YOLO; import matplotlib; print('laptop_python_packages_ok')" >> "%LOG_FILE%" 2>&1

if errorlevel 1 (
    echo [FAIL] Python import check failed.
    echo [FAIL] Python import check failed. >> "%LOG_FILE%"
    echo [HINT] Check logs\dependency_check.log.
    echo [HINT] Check logs\dependency_check.log. >> "%LOG_FILE%"
    exit /b 1
) else (
    echo [OK] Python imports are ready.
    echo [OK] Python imports are ready. >> "%LOG_FILE%"
)

echo. >> "%LOG_FILE%"

echo [TEST 5] Checking project files...
echo [TEST 5] Checking project files... >> "%LOG_FILE%"

if not exist scripts\laptop_ui_stream_control.py (
    echo [FAIL] Missing scripts\laptop_ui_stream_control.py
    echo [FAIL] Missing scripts\laptop_ui_stream_control.py >> "%LOG_FILE%"
    exit /b 1
)

if not exist models\validation.pt (
    echo [FAIL] Missing models\validation.pt
    echo [FAIL] Missing models\validation.pt >> "%LOG_FILE%"
    exit /b 1
)

if not exist models\anywheredoor\global_last.pt (
    echo [FAIL] Missing models\anywheredoor\global_last.pt
    echo [FAIL] Missing models\anywheredoor\global_last.pt >> "%LOG_FILE%"
    exit /b 1
)

if not exist models\anywheredoor\generator.pt (
    echo [FAIL] Missing models\anywheredoor\generator.pt
    echo [FAIL] Missing models\anywheredoor\generator.pt >> "%LOG_FILE%"
    exit /b 1
)

echo [OK] Required project files exist.
echo [OK] Required project files exist. >> "%LOG_FILE%"

echo. >> "%LOG_FILE%"
echo ======================================== >> "%LOG_FILE%"
echo Dependency Check Passed >> "%LOG_FILE%"
echo ======================================== >> "%LOG_FILE%"

echo [OK] Dependency check passed.
echo [INFO] Log saved to %LOG_FILE%

exit /b 0