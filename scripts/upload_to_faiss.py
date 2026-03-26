#!/usr/bin/env python3
"""
上传 README.md 到 FAISS 向量数据库
使用 RAG Pipeline 处理并存储文档到 FAISS
"""

import asyncio
import sys
from pathlib import Path

# 添加 src 到路径以便导入
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from src.aegis_isle.rag.pipeline import initialize_default_pipeline


async def upload_to_faiss():
    """上传 README.md 到 FAISS 向量数据库"""

    print("=" * 70)
    print("📦 FAISS 向量库上传脚本")
    print("=" * 70)
    print()

    # 文件路径
    readme_path = project_root / "README.md"

    # 检查文件是否存在
    if not readme_path.exists():
        print(f"❌ 错误: 未找到文件 {readme_path}")
        return False

    print(f"📄 文件路径: {readme_path}")
    print(f"📊 文件大小: {readme_path.stat().st_size} bytes")
    print()

    try:
        # 初始化 RAG Pipeline
        print("[1/4] 正在初始化 RAG Pipeline（包含 FAISS）...")
        pipeline = await initialize_default_pipeline()
        print("    ✅ Pipeline 初始化成功")

        # 验证 FAISS 已正确初始化
        if pipeline.retriever.vector_db_type != "faiss":
            print(f"    ⚠️  警告: 当前向量库类型为 {pipeline.retriever.vector_db_type}，不是 FAISS")
        else:
            print(f"    ✅ FAISS 向量库已就绪 (维度: {pipeline.retriever._dimension})")
            print(f"    📊 当前索引大小: {pipeline.retriever._vector_db.ntotal} 向量")
        print()

        # 上传文档到 FAISS
        print("[2/4] 正在处理并上传 README.md 到 FAISS...")
        print("    📝 步骤: 文档处理 → 分块 → 嵌入向量生成 → FAISS 存储")

        success = await pipeline.add_document(
            file_path=str(readme_path),
            metadata={
                "source": "project_documentation",
                "file_type": "markdown",
                "file_name": "README.md",
                "upload_time": "2026-02-04"
            }
        )

        if success:
            print("    ✅ 文档上传成功！")
        else:
            print("    ❌ 文档上传失败")
            return False
        print()

        # 验证存储成功
        print("[3/4] 验证 FAISS 存储...")
        faiss_size = pipeline.retriever._vector_db.ntotal
        chunk_count = len(pipeline.retriever._id_to_chunk)

        print(f"    📊 FAISS 索引大小: {faiss_size} 向量")
        print(f"    📊 文档分块映射: {chunk_count} 个 chunk")

        if faiss_size > 0 and chunk_count > 0:
            print("    ✅ FAISS 存储验证成功")
        else:
            print("    ⚠️  警告: FAISS 可能为空")
        print()

        # 测试检索功能
        print("[4/4] 测试 FAISS 检索功能...")
        test_query = "AegisIsle 项目介绍"
        print(f"    🔍 测试查询: '{test_query}'")

        search_result = await pipeline.retriever.search(
            query=test_query,
            limit=3
        )

        if search_result.results:
            print(f"    ✅ 检索成功，找到 {len(search_result.results)} 个相关文档")
            print()
            print("    📝 检索结果预览:")
            for i, result in enumerate(search_result.results[:2], 1):
                content_preview = result.chunk.content[:100].replace('\n', ' ')
                print(f"       [{i}] 相似度: {result.score:.4f}")
                print(f"           内容: {content_preview}...")
        else:
            print("    ⚠️  检索未找到结果（可能需要检查）")

        print()
        print("=" * 70)
        print("✅ 上传流程完成！")
        print("=" * 70)
        print()
        print("📊 总结:")
        print(f"   - FAISS 向量数: {faiss_size}")
        print(f"   - 文档分块数: {chunk_count}")
        print(f"   - 向量维度: {pipeline.retriever._dimension}")
        print(f"   - 检索测试: {'✅ 通过' if search_result.results else '⚠️ 未找到结果'}")
        print()

        return True

    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(upload_to_faiss())
    sys.exit(0 if result else 1)
