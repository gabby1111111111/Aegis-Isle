
import asyncio
import json
import logging
import sys
from unittest.mock import MagicMock, AsyncMock
from pathlib import Path

# 添加 src 到路径以便导入
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from src.aegis_isle.rag.pipeline import RAGPipeline, RAGConfig
from src.aegis_isle.rag.retriever import EnhancedQueryResult, RetrievalResult, DocumentChunk

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def mock_generate_stream(prompt, retrieval_context=None, **kwargs):
    """模拟 Generator 的流式输出"""
    chunks = ["这是", "一个", "模拟", "的", "流式", "回复", "。", "RAG", "流程", "工作", "正常", "！"]
    for chunk in chunks:
        await asyncio.sleep(0.1) # 模拟网络延迟
        yield chunk

async def test_stream_mock():
    """测试 Pipeline 的流式功能 (使用 Mock 组件)"""
    print("\n" + "="*50)
    print("开始测试 RAG Pipeline Query Stream (MOCK 模式)")
    print("="*50 + "\n")

    try:
        # 1. 创建 Mock Pipeline
        print("[1] 正在初始化 Mock Pipeline...")
        config = RAGConfig()
        pipeline = RAGPipeline(config)
        
        # Mock Retriever
        mock_retriever = AsyncMock()
        mock_chunk = DocumentChunk(
            document_id="test_doc_1", 
            content="RAG (Retrieval-Augmented Generation) 是一种结合了检索和生成的技术。", 
            chunk_index=0,
            metadata={"source": "wiki"}
        )
        mock_result = EnhancedQueryResult(
            query="test",
            results=[RetrievalResult(chunk=mock_chunk, score=0.95)],
            total_time=0.1
        )
        mock_retriever.search.return_value = mock_result
        pipeline.retriever = mock_retriever
        
        # Mock Generator
        mock_generator = MagicMock()
        mock_generator.generate_stream = mock_generate_stream
        pipeline.generator = mock_generator

        print("    Mock Pipeline 初始化成功")

        # 2. 执行流式查询
        query = "什么是 RAG？"
        print(f"\n[2] 开始执行查询: '{query}'")
        print("-" * 30 + " Stream Output " + "-" * 30)

        chunk_count = 0
        async for chunk in pipeline.query_stream(query):
            chunk_count += 1
            
            # 尝试解析是否为 Metadata JSON
            try:
                if chunk.strip().startswith('{') and '"type": "metadata"' in chunk:
                    meta = json.loads(chunk)
                    print(f"\n[METADATA RECEIVED]: {json.dumps(meta, indent=2, ensure_ascii=False)}\n")
                    print("-" * 30 + " Text Generation " + "-" * 30)
                    continue
            except:
                pass

            # 打印普通文本流
            print(f"'{chunk}'", end=" ", flush=True)

        print("\n\n" + "-" * 75)
        print(f"\n[3] 流式输出结束。共收到 {chunk_count} 个 chunks。")
        
        if chunk_count > 0:
            print("\n✅ 测试通过：成功接收到流式数据和元数据")
        else:
            print("\n❌ 测试失败：未收到任何数据")

    except Exception as e:
        print(f"\n\n[ERROR] 测试过程中发生错误:\n{str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_stream_mock())
