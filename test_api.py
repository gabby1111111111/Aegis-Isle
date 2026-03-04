"""测试 SiliconFlow API 配置"""

import asyncio
import os
from openai import AsyncOpenAI

async def test_api():
    """测试 API 连接"""
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")

    if not api_key:
        print("❌ 请先在 .env 中设置 OPENAI_API_KEY")
        return False

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    
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
