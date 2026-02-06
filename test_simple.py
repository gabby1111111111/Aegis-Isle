"""简化测试 - 单次请求验证"""

import requests
import json

BASE_URL = "http://127.0.0.1:8001"

def test_single_request():
    """测试单次聊天请求"""
    print("🧪 测试单次聊天请求\n")
    
    url = f"{BASE_URL}/v1/chat/completions"
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [
            {
                "role": "system",
                "content": "你是一个RPG游戏助手。当用户获得物品时，你需要用XML格式记录。"
            },
            {
                "role": "user",
                "content": "我在新手村买了一把木剑"
            }
        ],
        "stream": False,
        "user": "simple_test"
    }
    
    print(f"📤 发送请求到: {url}")
    print(f"📦 Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}\n")
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        
        print(f"📥 响应状态: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 请求成功!\n")
            print(f"AI 回复:")
            print(f"{data['choices'][0]['message']['content']}\n")
            print(f"Token 使用: {data.get('usage', {})}")
            return True
        else:
            print(f"❌ 请求失败")
            print(f"错误: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 异常: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_health():
    """检查服务器健康状态"""
    print("\n🏥 检查服务器健康状态\n")
    
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 服务器运行正常")
            print(f"版本: {data.get('version', 'N/A')}")
            print(f"文档: {data.get('docs', 'N/A')}")
            return True
        else:
            print(f"❌ 服务器响应异常: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 无法连接服务器: {e}")
        return False


if __name__ == "__main__":
    print("="*60)
    print("🚀 Aegis-Isle 简化测试")
    print("="*60)
    
    # 1. 检查服务器
    if not check_health():
        print("\n❌ 服务器未运行，请先启动服务器")
        exit(1)
    
    # 2. 测试聊天
    print("\n" + "="*60)
    success = test_single_request()
    print("="*60)
    
    if success:
        print("\n✅ 测试通过!")
    else:
        print("\n❌ 测试失败!")
