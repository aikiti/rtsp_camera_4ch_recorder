@echo off
setlocal

set APP=rtsp_4ch_recorder.py

if not exist "%APP%" (
  echo [ERROR] "%APP%" not found. Put this BAT next to %APP% (same folder) and run again.
  pause
  exit /b 1
)

echo [1/4] Upgrading pip...
python -m pip install --upgrade pip

echo [2/4] Installing requirements...
pip install -r requirements.txt
pip install pyinstaller

echo [3/4] Checking ffmpeg.exe...
if not exist "bin\ffmpeg.exe" (
  echo.
  echo [INFO] Please put "ffmpeg.exe" into ".\bin\" and run this BAT again.
  echo        You can get Windows builds from the official FFmpeg downloads page.
  echo        (Windows builds are provided by gyan.dev or BtbN.)
  echo.
  pause
)

echo [4/4] Building EXE with PyInstaller...
pyinstaller --noconfirm --clean ^
  --name "RTSP4CamRecorder" ^
  --windowed ^
  --collect-all PySide6 ^
  --collect-all cv2 ^
  --add-data "bin;bin" ^
  "%APP%"

echo.
echo [DONE] See ".\dist\RTSP4CamRecorder\RTSP4CamRecorder.exe"
echo        If SmartScreen warns (unsigned), click "More info" -> "Run anyway".
echo.
pause
