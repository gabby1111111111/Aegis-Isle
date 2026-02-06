"""测试 SiliconFlow API 配置"""

import asyncio
from openai import AsyncOpenAI

async def test_api():
    """测试 API 连接"""
    client = AsyncOpenAI(
        api_key="sk-enrrsvuvlvaztjmzilcxnofmowvttxsxydbosovlknmgqhar",
        base_url="https://api.siliconflow.cn/v1"
    )
    
    try:
        print("🔍 测试 SiliconFlow API 连接...")
        response = await client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[{"role": "user", "content": "你好"}],
            max_tokens=50
        )
        
        print("✅ API 连接成功!")
        print(f"模型: {response.model}")
        print(f"回复: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ API 连接失败: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_api())
