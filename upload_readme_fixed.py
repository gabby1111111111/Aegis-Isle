#!/usr/bin/env python3
"""
通过文本 API 上传 README.md 到 FAISS（修复版）
"""

import requests
import json
from pathlib import Path

def upload_readme():
    """通过文本 API 上传 README"""
    
    readme_path = Path("README.md")
    
    if not readme_path.exists():
        print(f"❌ 错误: 未找到文件 {readme_path}")
        return False
    
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    url = "http://localhost:8002/api/v1/documents/text"
    
    # 使用空 metadata 避免与 DocumentMetadata 构造函数参数冲突
    payload = {
        "content": content,
        "metadata": {}
    }
    
    print("=" * 70)
    print("📤 通过文本 API 上传 README.md（修复版）")
    print("=" * 70)
    print()
    print(f"📄 文件: {readme_path}")
    print(f"📊 内容长度: {len(content)} 字符")
    print(f"🌐 API: {url}")
    print()
    print("⏳ 正在上传（metadata已修复）...")
    
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
            
            # 检查 FAISS 状态
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
                    
                    # 测试检索
                    print("\n" + "=" * 70)
                    print("🔍 测试 RAG 检索")
                    print("=" * 70)
                    query_url = "http://localhost:8002/v1/chat/completions"
                    test_payload = {
                        "model": "Qwen/Qwen2.5-7B-Instruct",
                        "messages": [{"role": "user", "content": "什么是 AegisIsle？"}],
                        "stream": False
                    }
                    query_resp = requests.post(query_url, json=test_payload, timeout=30)
                    if query_resp.status_code == 200:
                        result = query_resp.json()
                        answer = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                        print(f"AI 回答: {answer[:200]}...\n")
                    
                    return True
                else:
                    print("⚠️  警告: FAISS 仍然为空")
                    return False
            else:
                print("⚠️  无法获取 FAISS 状态")
                return True
                
        else:
            print(f"❌ 上传失败")
            print(f"错误信息: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = upload_readme()
    exit(0 if success else 1)
