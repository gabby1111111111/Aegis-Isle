#!/usr/bin/env python3
"""
测试服务器的 RAG 检索功能
"""

import requests
import json

def test_rag_query():
    """测试 RAG 查询"""
    
    url = "http://localhost:8002/v1/chat/completions"
    
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": "介绍一下 Aegis-Isle 项目"
            }
        ],
        "stream": False
    }
    
    print("=" * 60)
    print("🔍 测试 RAG 查询功能")
    print("=" * 60)
    print()
    print(f"📝 查询: {payload['messages'][0]['content']}")
    print(f"🌐 API: {url}")
    print()
    print("⏳ 正在查询...")
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        
        print(f"📊 HTTP 状态码: {response.status_code}")
        print()
        
        if response.status_code == 200:
            data = response.json()
            
            print("✅ 查询成功！")
            print()
            
            # 提取响应
            if 'choices' in data and len(data['choices']) > 0:
                message = data['choices'][0].get('message', {})
                content = message.get('content', '')
                
                print("📋 AI 响应:")
                print(f"{content[:500]}...")  # 显示前500字符
                print()
                
            # 检查是否使用了 RAG
            if 'usage' in data:
                print("📊 使用统计:")
                print(f"   {json.dumps(data['usage'], indent=2, ensure_ascii=False)}")
                print()
                
            return True
        else:
            print(f"❌ 查询失败")
            print(f"错误信息: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_rag_query()
    exit(0 if success else 1)
