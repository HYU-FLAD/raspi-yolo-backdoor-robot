@echo off
setlocal

set PI_USER=fl
set PI_IP=192.168.24.50

ssh %PI_USER%@%PI_IP% "pkill -f '[p]i_camera2_pub.py' 2>/dev/null || true; pkill -f '[m]otor_zmq_server.py' 2>/dev/null || true"

pause