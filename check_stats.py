#!/usr/bin/env python3
"""
查看服务器文档统计
"""

import requests
import json

def get_stats():
    """获取文档统计"""
    
    url = "http://localhost:8002/api/v1/documents/stats"
    
    print("=" * 60)
    print("📊 服务器文档统计")
    print("=" * 60)
    print()
    
    try:
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ 获取成功！")
            print()
            print(json.dumps(data, indent=2, ensure_ascii=False))
            print()
            
            # 重点显示
            if 'retriever_stats' in data:
                stats = data['retriever_stats']
                print("📋 关键信息:")
                print(f"   - 总文档数: {stats.get('total_documents', 'N/A')}")
                print(f"   - 总分块数: {stats.get('total_chunks', 'N/A')}")
                print(f"   - 向量维度: {stats.get('vector_dimension', 'N/A')}")
            
            return True
        else:
            print(f"❌ 获取失败: HTTP {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        return False

if __name__ == "__main__":
    get_stats()
