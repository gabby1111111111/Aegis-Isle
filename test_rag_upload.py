#!/usr/bin/env python3
"""
测试RAG文档上传功能
"""

import asyncio
import sys
sys.path.insert(0, 'src')

from aegis_isle.rag.pipeline import RAGPipeline, RAGConfig
from aegis_isle.rag.document_processor import DocumentProcessor
from aegis_isle.core.config import settings

async def test_rag_upload():
    """测试RAG文档上传"""

    print("=" * 50)
    print("开始测试RAG文档上传")
    print("=" * 50)

    # 创建RAG配置
    config = RAGConfig(
        embedding_model=settings.embedding_model,
        vector_db_type=settings.vector_db_type,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )

    print(f"\n配置信息:")
    print(f"  嵌入模型: {config.embedding_model}")
    print(f"  向量数据库: {config.vector_db_type}")
    print(f"  块大小: {config.chunk_size}")
    print(f"  块重叠: {config.chunk_overlap}")

    # 创建RAG管道
    print("\n初始化RAG管道...")
    pipeline = RAGPipeline(config)

    # 读取测试文件
    test_file = "rag_test_data.txt"
    print(f"\n读取测试文件: {test_file}")

    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"  文件大小: {len(content)} 字符")
        print(f"  前100字符: {content[:100]}...")
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {test_file}")
        return

    # 处理文档
    print("\n处理文档...")
    processor = DocumentProcessor()
    document = processor.process_text(
        content=content,
        document_id="test_doc_1",
        metadata={"source": test_file}
    )

    print(f"  文档ID: {document.id}")
    print(f"  文档类型: {document.content_type}")
    print(f"  元数据: {document.metadata}")

    # 添加到RAG管道
    print("\n添加文档到RAG管道...")
    try:
        result = await pipeline.add_document(document)
        if result:
            print("✅ 文档添加成功!")

            # 测试搜索
            print("\n测试搜索功能...")
            query = "测试查询"
            print(f"  查询: {query}")

            search_result = await pipeline.query(query, top_k=3)
            print(f"  找到 {len(search_result.retrieval_result.results)} 个相关结果")

            for i, result in enumerate(search_result.retrieval_result.results[:3], 1):
                print(f"\n  结果 #{i}:")
                print(f"    得分: {result.score:.4f}")
                print(f"    内容: {result.content[:100]}...")
        else:
            print("❌ 文档添加失败")

    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_rag_upload())
