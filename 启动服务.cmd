@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv312\Scripts\python.exe" (
  echo [首次运行] 未找到 .venv312，正在创建 Python 3.12 虚拟环境...
  py -3.12 -m venv .venv312 2>nul
  if not exist ".venv312\Scripts\python.exe" (
    python -m venv .venv312
  )
  if not exist ".venv312\Scripts\python.exe" (
    echo 创建虚拟环境失败，请先安装 Python 3.12 并加入 PATH。
    pause
    exit /b 1
  )
  ".venv312\Scripts\python.exe" -m pip install --upgrade pip
  ".venv312\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo 依赖安装失败，请检查网络后重试。
    pause
    exit /b 1
  )
)

".venv312\Scripts\python.exe" "启动本地服务.py" --background
if errorlevel 1 pause
