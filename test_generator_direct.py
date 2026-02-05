"""
测试 LLM Generator 直接调用
"""
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from src.aegis_isle.rag.generator import get_generator
from src.aegis_isle.core.config import settings

async def test_generator():
    print("="*60)
    print("测试 LLM Generator")
    print("="*60)
    print(f"Provider: {settings.llm_provider}")
    print(f"Model: {settings.default_llm_model}")
    print(f"Base URL: {settings.openai_base_url}")
    print(f"API Key: {settings.openai_api_key[:20]}...")
    print("="*60 + "\n")
    
    # 初始化 Generator
    generator = get_generator()
    
    # 测试普通生成
    print("📝 测试普通生成:")
    query = "你好，请用中文介绍一下你自己"
    print(f"Query: {query}\n")
    
    result = await generator.generate(query)
    print(f"✅ Response: {result.generated_text}\n")
    print(f"Model: {result.model}")
    print(f"Time: {result.generation_time:.2f}s\n")
    
    # 测试流式生成
    print("="*60)
    print("📡 测试流式生成:")
    print("="*60)
    
    chunk_count = 0
    print("Stream: ", end="", flush=True)
    async for chunk in generator.generate_stream(query):
        print(chunk, end="", flush=True)
        chunk_count += 1
    
    print(f"\n\n✅ 流式生成完成，共 {chunk_count} 个chunks")

if __name__ == "__main__":
    asyncio.run(test_generator())
