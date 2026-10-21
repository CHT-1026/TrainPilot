@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv312\Scripts\python.exe" (
  echo 未找到 .venv312，请先运行 启动服务.cmd。
  pause
  exit /b 1
)
".venv312\Scripts\python.exe" -m enterprise_rag.eval.run_eval --gate
pause
