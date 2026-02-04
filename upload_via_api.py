#!/usr/bin/env python3
"""
通过服务器 API 上传 README.md 到 FAISS
"""

import requests
from pathlib import Path

def upload_via_api():
    """通过服务器 API 上传文档"""
    
    # 文件路径
    readme_path = Path("README.md")
    
    if not readme_path.exists():
        print(f"❌ 错误: 未找到文件 {readme_path}")
        return False
    
    # API 端点
    url = "http://localhost:8002/api/v1/documents/upload"
    
    print("=" * 60)
    print("📤 通过服务器 API 上传 README.md")
    print("=" * 60)
    print()
    print(f"📄 文件: {readme_path}")
    print(f"🌐 API: {url}")
    print()
    print("⏳ 正在上传...")
    
    try:
        # 打开并上传文件
        with open(readme_path, 'rb') as f:
            files = {'file': (readme_path.name, f, 'text/markdown')}
            response = requests.post(url, files=files, timeout=60)
        
        print(f"📊 HTTP 状态码: {response.status_code}")
        print()
        
        if response.status_code == 200:
            data = response.json()
            print("✅ 上传成功！")
            print()
            print("📋 响应数据:")
            print(f"   Success: {data.get('success')}")
            print(f"   Message: {data.get('message')}")
            if 'metadata' in data:
                print(f"   文件名: {data['metadata'].get('filename')}")
                print(f"   文件大小: {data['metadata'].get('file_size')} bytes")
            print()
            return True
        else:
            print(f"❌ 上传失败")
            print(f"错误信息: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ 错误: 无法连接到服务器")
        print("请确认服务器正在运行在 http://localhost:8002")
        return False
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        return False

if __name__ == "__main__":
    success = upload_via_api()
    exit(0 if success else 1)
