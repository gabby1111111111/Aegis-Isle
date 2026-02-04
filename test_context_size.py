"""
测试带长上下文的请求，模拟 SillyTavern 的场景
"""
import asyncio
import aiohttp
import json

async def test_long_context():
    url = "http://127.0.0.1:8002/v1/chat/completions"
    
    # 模拟一个较长的对话历史
    long_message = "请详细介绍一下什么是 RAG（Retrieval-Augmented Generation）？"
    
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [
            {"role": "system", "content": "你是一个专业的 AI 助手。"},
            {"role": "user", "content": long_message}
        ],
        "stream": True,
        "max_tokens": 2048,
        "temperature": 0.7
    }
    
    print(f"📤 发送请求...")
    print(f"Messages: {len(payload['messages'])} 条")
    print(f"Total chars: {sum(len(m['content']) for m in payload['messages'])}")
    print("\n" + "="*60)
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as response:
                print(f"Status: {response.status}")
                
                if response.status != 200:
                    error_text = await response.text()
                    print(f"❌ 错误响应:\n{error_text}")
                    return
                
                print("✅ 开始接收流式响应:\n")
                
                chunk_count = 0
                async for line in response.content:
                    line_str = line.decode('utf-8').strip()
                    
                    if not line_str or line_str == "data: [DONE]":
                        continue
                    
                    if line_str.startswith("data: "):
                        try:
                            json_str = line_str[6:]
                            chunk_data = json.loads(json_str)
                            
                            if chunk_data.get("choices"):
                                delta = chunk_data["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    print(content, end="", flush=True)
                                    chunk_count += 1
                        except Exception as e:
                            print(f"\n解析错误: {e}")
                            print(f"Raw line: {line_str[:200]}")
                
                print(f"\n\n✅ 完成！共 {chunk_count} 个 chunks")
    
    except asyncio.TimeoutError:
        print("❌ 请求超时")
    except Exception as e:
        print(f"❌ 异常: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(test_long_context())
