#!/usr/bin/env python3
"""
上传 README.md 到向量数据库
使用 RAG Pipeline 处理并存储文档
"""

import asyncio
import sys
from pathlib import Path

# 添加 src 到路径以便导入
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from src.aegis_isle.rag.pipeline import initialize_default_pipeline


async def upload_readme():
    """上传 README.md 到向量数据库"""
    
    print("=" * 60)
    print("README.md 上传脚本")
    print("=" * 60)
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
        print("[1/3] 正在初始化 RAG Pipeline...")
        pipeline = await initialize_default_pipeline()
        print("    ✅ Pipeline 初始化成功")
        print()
        
        # 上传文档
        print("[2/3] 正在上传 README.md 到向量数据库...")
        success = await pipeline.add_document(
            file_path=str(readme_path),
            metadata={
                "source": "project_documentation",
                "file_type": "markdown",
                "file_name": "README.md"
            }
        )
        
        if success:
            print("    ✅ 文档上传成功")
            print()
            
            # 获取统计信息
            print("[3/3] 获取系统统计信息...")
            stats = await pipeline.get_stats()
            
            if "retriever_stats" in stats:
                retriever_stats = stats["retriever_stats"]
                print(f"    📊 向量数据库状态:")
                print(f"       - 总文档数: {retriever_stats.get('total_documents', 'N/A')}")
                print(f"       - 总分块数: {retriever_stats.get('total_chunks', 'N/A')}")
            
            print()
            print("=" * 60)
            print("✅ 上传完成！")
            print("=" * 60)
            return True
        else:
            print("    ❌ 文档上传失败")
            return False
            
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(upload_readme())
    sys.exit(0 if result else 1)
