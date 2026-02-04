"""
测试 API 并捕获完整响应内容
"""
import asyncio
import aiohttp
import json

async def test_with_question(question):
    url = "http://127.0.0.1:8002/v1/chat/completions"
    
    payload = {
        "model": "rag-default",
        "messages": [
            {"role": "user", "content": question}
        ],
        "stream": True
    }
    
    print(f"\n{'='*60}")
    print(f"📤 问题: {question}")
    print('='*60)
    
    full_response = ""
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                print(f"❌ 错误: {response.status}")
                error_text = await response.text()
                print(error_text)
                return
            
            async for line in response.content:
                line_str = line.decode('utf-8').strip()
                if not line_str or line_str == "data: [DONE]":
                    continue
                
                if line_str.startswith("data: "):
                    try:
                        json_str = line_str[6:]  # Remove "data: " prefix
                        chunk_data = json.loads(json_str)
                        
                        # Extract content from delta
                        if chunk_data.get("choices"):
                            delta = chunk_data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                full_response += content
                                print(content, end="", flush=True)
                    except:
                        pass
    
    print(f"\n\n{'='*60}")
    print(f"✅ 完整回复: {full_response}")
    print(f"📊 长度: {len(full_response)} 字符")
    print('='*60)

async def main():
    # 测试多个问题
    questions = [
        "hi",
        "介绍一下什么是RAG",
        "你好，请用中文回答"
    ]
    
    for q in questions:
        await test_with_question(q)
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
