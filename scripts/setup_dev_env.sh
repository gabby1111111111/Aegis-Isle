#!/bin/bash

echo "==================================================="
echo "   AegisIsle 本地开发环境设置脚本 (Linux/Mac)"
echo "==================================================="
echo

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: Python 3 未安装"
    echo "请先安装 Python 3.9+"
    exit 1
fi

echo "✅ 检测到 Python:"
python3 --version

echo
echo "📂 创建虚拟环境..."
if [ -d "venv" ]; then
    echo "⚠️  虚拟环境已存在，跳过创建"
else
    python3 -m venv venv
    echo "✅ 虚拟环境创建完成"
fi

echo
echo "🔄 激活虚拟环境..."
source venv/bin/activate

echo
echo "📦 安装项目依赖..."
python -m pip install --upgrade pip
pip install -r requirements.txt

echo
echo "📁 创建必要的目录..."
mkdir -p logs/{audit,application,errors}
mkdir -p data
mkdir -p uploads

echo
echo "⚙️  检查配置文件..."
if [ ! -f ".env" ]; then
    echo "📋 复制环境配置文件..."
    cp .env.example .env
    echo "✅ 已创建 .env 文件，请根据需要修改配置"
else
    echo "✅ .env 文件已存在"
fi

echo
echo "==================================================="
echo "🎉 开发环境设置完成！"
echo "==================================================="
echo
echo "📖 使用指南:"
echo "1. 激活虚拟环境: source venv/bin/activate"
echo "2. 启动开发服务: python run_dev.py"
echo "3. 或直接使用: uvicorn src.aegis_isle.api.main:app --reload --host 0.0.0.0 --port 8000"
echo
echo "🌐 访问地址:"
echo "  - API文档: http://localhost:8000/docs"
echo "  - ReDoc:   http://localhost:8000/redoc"
echo "  - 根端点: http://localhost:8000/"
echo
echo "👥 默认账户:"
echo "  - 管理员: admin / admin123"
echo "  - 普通用户: testuser / testpass123"
echo