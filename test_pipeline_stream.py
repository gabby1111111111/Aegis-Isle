
import asyncio
import json
import logging
import sys
from pathlib import Path

# 添加 src 到路径以便导入
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from src.aegis_isle.rag.pipeline import initialize_default_pipeline
from src.aegis_isle.core.config import settings

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_stream():
    """测试 Pipeline 的流式功能"""
    print("\n" + "="*50)
    print("开始测试 RAG Pipeline Query Stream")
    print("="*50 + "\n")

    try:
        # 1. 初始化 Pipeline
        print("[1] 正在初始化 RAG Pipeline...")
        pipeline = await initialize_default_pipeline()
        print("    Pipeline 初始化成功")

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

            # 打印普通文本流，不换行
            print(chunk, end="", flush=True)

        print("\n" + "-" * 75)
        print(f"\n[3] 流式输出结束。共收到 {chunk_count} 个 chunks。")

    except Exception as e:
        print(f"\n\n[ERROR] 测试过程中发生错误:\n{str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_stream())
