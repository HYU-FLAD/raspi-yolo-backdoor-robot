@echo off
setlocal

cd /d "%~dp0.."

echo [INFO] Project root:
cd

echo [INFO] Running AnyWhereDoor trigger stream...
echo [INFO] Camera: tcp://192.168.24.50:5556
echo [INFO] Motor : tcp://192.168.24.50:5555

python scripts\laptop_sun_backdoor_stream_send_anywheredoor.py ^
  --clean-model models\validation.pt ^
  --model models\anywheredoor\global_best.pt ^
  --generator models\anywheredoor\generator.pt ^
  --camera tcp://192.168.24.50:5556 ^
  --pi tcp://192.168.24.50:5555

pause
