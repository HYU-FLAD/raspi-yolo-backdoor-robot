@echo off
setlocal

cd /d "%~dp0.."

echo [INFO] Project root:
cd

echo [INFO] Running laptop UI stream control...
echo [INFO] Camera: tcp://192.168.24.50:5556
echo [INFO] Motor : tcp://192.168.24.50:5555

python scripts\laptop_ui_stream_control.py ^
  --camera tcp://192.168.24.50:5556 ^
  --pi tcp://192.168.24.50:5555

pause
