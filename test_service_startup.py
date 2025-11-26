#!/usr/bin/env python3
"""
测试完整服务启动（不实际启动服务器）
验证所有模块可以正确导入和初始化
"""

import sys
sys.path.insert(0, 'src')

def test_imports():
    """测试关键模块导入"""
    print("=" * 50)
    print("测试模块导入")
    print("=" * 50)

    try:
        print("1. 测试核心配置...")
        from aegis_isle.core.config import settings
        print("✅ 核心配置导入成功")

        print("2. 测试日志系统...")
        from aegis_isle.core.logging import logger, audit_logger
        print("✅ 日志系统导入成功")

        print("3. 测试RAG组件...")
        from aegis_isle.rag.pipeline import RAGPipeline, RAGConfig
        from aegis_isle.rag.document_processor import DocumentProcessor
        from aegis_isle.rag.retriever import LegacyRetriever
        from aegis_isle.rag.generator import TextGenerator
        print("✅ RAG组件导入成功")

        print("4. 测试Agent系统...")
        from aegis_isle.agents.base import BaseAgent, AgentConfig, AgentRole
        from aegis_isle.agents.orchestrator import ToolIntegratedOrchestrator
        print("✅ Agent系统导入成功")

        print("5. 测试工具系统...")
        from aegis_isle.tools import get_tool_registry
        print("✅ 工具系统导入成功")

        print("\n" + "=" * 50)
        print("✅ 所有关键模块导入成功！")
        print("=" * 50)
        return True

    except Exception as e:
        print(f"❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_initialization():
    """测试基本初始化"""
    print("\n" + "=" * 50)
    print("测试基本初始化")
    print("=" * 50)

    try:
        from aegis_isle.core.config import settings

        print(f"Environment: {settings.environment}")
        print(f"Debug: {settings.debug}")
        print(f"LLM Provider: {settings.llm_provider}")
        print(f"Embedding Model: {settings.embedding_model}")
        print(f"Vector DB Type: {settings.vector_db_type}")

        if settings.openai_base_url:
            print(f"Custom OpenAI Base URL: {settings.openai_base_url}")

        print("\n✅ 基本配置读取成功！")
        return True

    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print("AegisIsle 服务启动前预检查")
    print("=" * 60)

    # 测试导入
    if not test_imports():
        print("\n❌ 模块导入测试失败")
        return False

    # 测试初始化
    if not test_basic_initialization():
        print("\n❌ 基本初始化测试失败")
        return False

    print("\n" + "=" * 60)
    print("🎉 所有预检查通过！服务应该可以正常启动")
    print("=" * 60)
    print("\n💡 建议:")
    print("1. 现在可以启动完整服务: python run_dev.py --mode full")
    print("2. 或启动认证服务: python run_dev.py --mode auth")
    print("3. 如果看到任何错误，请检查 .env 配置")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)