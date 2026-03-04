@echo off
echo ===================================================
echo    AegisIsle 本地开发环境设置脚本 (Windows)
echo ===================================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ❌ 错误: Python 未安装或未添加到 PATH
    echo 请先安装 Python 3.9+ 并添加到系统 PATH
    pause
    exit /b 1
)

echo ✅ 检测到 Python:
python --version

echo.
echo 📂 创建虚拟环境...
if exist venv (
    echo ⚠️  虚拟环境已存在，跳过创建
) else (
    python -m venv venv
    echo ✅ 虚拟环境创建完成
)

echo.
echo 🔄 激活虚拟环境...
call venv\Scripts\activate.bat

echo.
echo 📦 安装项目依赖...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo 📁 创建必要的目录...
if not exist logs mkdir logs
if not exist logs\audit mkdir logs\audit
if not exist logs\application mkdir logs\application
if not exist logs\errors mkdir logs\errors
if not exist data mkdir data
if not exist uploads mkdir uploads

echo.
echo ⚙️  检查配置文件...
if not exist .env (
    echo 📋 复制环境配置文件...
    copy .env.example .env
    echo ✅ 已创建 .env 文件，请根据需要修改配置
) else (
    echo ✅ .env 文件已存在
)

echo.
echo ===================================================
echo 🎉 开发环境设置完成！
echo ===================================================
echo.
echo 📖 使用指南:
echo 1. 激活虚拟环境: venv\Scripts\activate
echo 2. 启动开发服务: python run_dev.py
echo 3. 或直接使用: uvicorn src.aegis_isle.api.main:app --reload --host 0.0.0.0 --port 8000
echo.
echo 🌐 访问地址:
echo   - API文档: http://localhost:8000/docs
echo   - ReDoc:   http://localhost:8000/redoc
echo   - 根端点: http://localhost:8000/
echo.
echo 👥 默认账户:
echo   - 管理员: admin / admin123
echo   - 普通用户: testuser / testpass123
echo.
pause