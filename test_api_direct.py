"""
直接测试 /v1/chat/completions API 端点
"""
import asyncio
import aiohttp
import json

async def test_chat_completions():
    url = "http://127.0.0.1:8002/v1/chat/completions"
    
    payload = {
        "model": "rag-default",
        "messages": [
            {"role": "user", "content": "你好，请介绍一下你自己"}
        ],
        "stream": True
    }
    
    print(f"📡 发送请求到: {url}")
    print(f"📦 Payload: {json.dumps(payload, ensure_ascii=False, indent=2)}")
    print("\n" + "="*60)
    print("📥 流式响应:")
    print("="*60 + "\n")
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            print(f"Status: {response.status}")
            print(f"Headers: {dict(response.headers)}\n")
            
            if response.status != 200:
                error_text = await response.text()
                print(f"❌ 错误: {error_text}")
                return
            
            # 读取流式响应
            chunk_count = 0
            async for line in response.content:
                line_str = line.decode('utf-8').strip()
                if line_str:
                    print(f"[Chunk {chunk_count}]: {line_str}")
                    chunk_count += 1
            
            if chunk_count == 0:
                print("⚠️  警告：没有收到任何数据块！")
            else:
                print(f"\n✅ 共收到 {chunk_count} 个数据块")

if __name__ == "__main__":
    asyncio.run(test_chat_completions())
