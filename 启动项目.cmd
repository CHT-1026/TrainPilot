@echo off
chcp 65001 >nul
cd /d "%~dp0"
"%~dp0.venv-agent\Scripts\python.exe" -u -m agent %*
if errorlevel 1 pause
