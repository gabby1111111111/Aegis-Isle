#!/usr/bin/env python3
"""
通过文本 API 上传 README.md 到 FAISS
"""

import requests
import json
from pathlib import Path

def upload_via_text_api():
    """通过文本 API 上传文档"""
    
    # 读取 README.md
    readme_path = Path("README.md")
    
    if not readme_path.exists():
        print(f"❌ 错误: 未找到文件 {readme_path}")
        return False
    
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # API 端点
    url = "http://localhost:8002/api/v1/documents/text"
    
    payload = {
        "content": content,
        "metadata": {
            "document_source": "README.md",
            "file_type": "markdown",
            "upload_method": "text_api"
        }
    }
    
    print("=" * 70)
    print("📤 通过文本 API 上传 README.md")
    print("=" * 70)
    print()
    print(f"📄 文件: {readme_path}")
    print(f"📊 内容长度: {len(content)} 字符")
    print(f"🌐 API: {url}")
    print()
    print("⏳ 正在上传...")
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        
        print(f"📊 HTTP 状态码: {response.status_code}")
        print()
        
        if response.status_code == 200:
            data = response.json()
            print("✅ 上传成功！")
            print()
            print("📋 响应数据:")
            print(json.dumps(data, indent=2, ensure_ascii=False))
            print()
            
            # 立即检查 FAISS 状态
            print("=" * 70)
            print("🔍 验证 FAISS 状态")
            print("=" * 70)
            print()
            
            stats_response = requests.get("http://localhost:8002/api/v1/documents/stats", timeout=10)
            if stats_response.status_code == 200:
                stats = stats_response.json()
                retriever_stats = stats.get('stats', {}).get('retriever_stats', {})
                
                total_chunks = retriever_stats.get('total_chunks', 0)
                print(f"📊 FAISS 状态:")
                print(f"   - 总分块数: {total_chunks}")
                print(f"   - 向量维度: {retriever_stats.get('vector_dimension', 'N/A')}")
                print()
                
                if total_chunks > 0:
                    print("✅ FAISS 中已有文档！")
                    return True
                else:
                    print("⚠️  警告: FAISS 仍然为空")
                    return False
            else:
                print("⚠️  无法获取 FAISS 状态")
                return True  # 上传成功但无法验证
                
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
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = upload_via_text_api()
    exit(0 if success else 1)
