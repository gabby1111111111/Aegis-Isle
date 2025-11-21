#!/usr/bin/env python3
"""
AegisIsle 开发服务器启动脚本
提供便捷的开发环境启动方式
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# 设置控制台编码为UTF-8 (Windows兼容)
if sys.platform.startswith('win'):
    import locale
    try:
        # 尝试设置UTF-8编码
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        # 如果失败，禁用emoji
        pass

def check_venv():
    """检查虚拟环境是否激活"""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        return True
    return False

def check_dependencies():
    """检查关键依赖是否安装"""
    try:
        import fastapi
        import uvicorn
        import pydantic
        return True
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请运行: pip install -r requirements.txt")
        return False

def ensure_directories():
    """确保必要的目录存在"""
    dirs = [
        "logs/audit",
        "logs/application",
        "logs/errors",
        "data",
        "uploads"
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    print("✅ 目录结构检查完成")

def check_env_file():
    """检查环境配置文件"""
    if not Path(".env").exists():
        if Path(".env.example").exists():
            print("📋 复制 .env.example 到 .env...")
            import shutil
            shutil.copy(".env.example", ".env")
            print("✅ 已创建 .env 文件")
        else:
            print("⚠️  未找到 .env 或 .env.example 文件")
    else:
        print("✅ .env 文件存在")

def safe_print(text):
    """安全打印，处理编码问题"""
    try:
        print(text)
    except UnicodeEncodeError:
        # 移除emoji和特殊字符，使用ASCII版本
        ascii_text = text.encode('ascii', 'ignore').decode('ascii')
        print(ascii_text)

def start_server(mode="full", host="0.0.0.0", port=8000, reload=True):
    """启动开发服务器"""

    safe_print("=================================================")
    safe_print("🚀 启动 AegisIsle 开发服务器")
    safe_print("=================================================")

    # 环境检查
    safe_print("🔍 环境检查...")

    if not check_venv():
        safe_print("⚠️  未检测到虚拟环境，建议在虚拟环境中运行")

    if not check_dependencies():
        sys.exit(1)

    ensure_directories()
    check_env_file()

    safe_print("✅ 环境检查完成")
    safe_print("")

    # 根据模式选择启动方式
    if mode == "auth":
        safe_print("🔐 启动简化认证服务器...")
        app_module = "auth_server_simple:app"
        safe_print("📝 注意: 这是简化版本，仅包含OAuth2+RBAC+审计日志功能")
    else:
        safe_print("🌟 启动完整AegisIsle服务器...")
        app_module = "src.aegis_isle.api.main:app"
        safe_print("📝 注意: 完整版本包含RAG、Agent、Tools等所有功能")

    safe_print(f"🌐 服务器地址: http://{host}:{port}")
    safe_print("📖 API文档: http://localhost:8000/docs")
    safe_print("📚 ReDoc: http://localhost:8000/redoc")
    safe_print("")
    safe_print("👥 默认账户:")
    safe_print("   - 管理员: admin / admin123")
    safe_print("   - 普通用户: testuser / testpass123")
    safe_print("")
    safe_print("🛑 按 Ctrl+C 停止服务器")
    safe_print("=================================================")
    safe_print("")

    # 启动uvicorn
    cmd = [
        "uvicorn",
        app_module,
        "--host", host,
        "--port", str(port),
        "--log-level", "info"
    ]

    if reload:
        cmd.append("--reload")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        safe_print("\n🛑 服务器已停止")
    except FileNotFoundError:
        safe_print("❌ 错误: 未找到 uvicorn")
        safe_print("请安装: pip install uvicorn")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        safe_print(f"❌ 服务器启动失败: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="AegisIsle 开发服务器")
    parser.add_argument(
        "--mode",
        choices=["full", "auth"],
        default="auth",
        help="启动模式: full(完整版) 或 auth(简化认证版，默认)"
    )
    parser.add_argument("--host", default="0.0.0.0", help="绑定主机 (默认: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="端口号 (默认: 8000)")
    parser.add_argument("--no-reload", action="store_true", help="禁用自动重载")

    args = parser.parse_args()

    start_server(
        mode=args.mode,
        host=args.host,
        port=args.port,
        reload=not args.no_reload
    )

if __name__ == "__main__":
    main()