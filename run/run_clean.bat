@echo off
setlocal

cd /d "%~dp0.."

echo [INFO] Project root:
cd

echo [INFO] Running clean model stream...
echo [INFO] Camera: tcp://192.168.24.50:5556
echo [INFO] Motor : tcp://192.168.24.50:5555

python scripts\laptop_sun_backdoor_stream_send.py ^
  --model models\oda\clean.pt ^
  --clean-model models\oda\clean.pt ^
  --camera tcp://192.168.24.50:5556 ^
  --pi tcp://192.168.24.50:5555 ^
  --no-trigger

pause
