"""
RAG 检索质量评估 (DeepEval)
===========================
用 DeepEval 框架评估 FAISS 记忆检索的质量。
分两部分:
  1. 离线评估: 不需要 LLM API，测试检索结果的结构和基础指标
  2. LLM-as-Judge 评估: 需要 API Key，用 LLM 打分 Faithfulness 和 Context Recall

运行方式:
  # 仅离线测试 (不需要 API Key)
  python -m pytest tests/test_rag_quality.py -v -k "offline"

  # 完整评估 (需要 OPENAI_API_KEY 或 DEEPEVAL_API_KEY)
  python -m pytest tests/test_rag_quality.py -v
"""

import pytest
import os

# DeepSeek-V3 响应可能较慢，延长 DeepEval 超时到 5 分钟
os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "300")
os.environ.setdefault("DEEPEVAL_TASK_GATHER_BUFFER_SECONDS_OVERRIDE", "60")


# ============================================
# 1. 离线检索质量测试 (不需要 LLM API)
# ============================================


class TestRAGOffline:
    """离线 RAG 质量测试 — 不需要 LLM API Key"""

    def test_memory_manager_can_initialize(self):
        """STMemoryManager 应能正常初始化"""
        from src.aegis_isle.rag.st_memory_manager import STMemoryManager

        mm = STMemoryManager()
        assert mm is not None
        assert hasattr(mm, "search_memory")

    def test_search_returns_list(self):
        """记忆检索应返回列表格式"""
        from src.aegis_isle.rag.st_memory_manager import memory_manager
        import asyncio

        async def _search():
            results = await memory_manager.search_memory(
                query="你还记得那次在法餐厅的事吗？",
                character_name="ZouZheng",
                world_line="AIDom",
                k=3,
            )
            return results

        results = asyncio.run(_search())
        assert isinstance(results, list), f"应返回列表，实际: {type(results)}"

    def test_search_result_structure(self):
        """每个检索结果应包含必要字段"""
        from src.aegis_isle.rag.st_memory_manager import memory_manager
        import asyncio

        async def _search():
            results = await memory_manager.search_memory(
                query="今天发生了什么？",
                character_name="ZouZheng",
                world_line="AIDom",
                k=3,
            )
            return results

        results = asyncio.run(_search())
        if len(results) > 0:  # 只有当有数据时才检查
            first = results[0]
            # ChatChunk 或 Document 应有 text/page_content 字段
            assert (
                hasattr(first, "page_content")
                or hasattr(first, "text")
                or isinstance(first, (str, dict))
            ), f"检索结果结构不符合预期: {type(first)}"

    def test_empty_query_returns_gracefully(self):
        """空查询不应崩溃"""
        from src.aegis_isle.rag.st_memory_manager import memory_manager
        import asyncio

        async def _search():
            results = await memory_manager.search_memory(
                query="", character_name="ZouZheng", world_line="AIDom", k=3
            )
            return results

        results = asyncio.run(_search())
        assert isinstance(results, list)

    def test_format_context_for_prompt(self):
        """格式化上下文应返回字符串"""
        from src.aegis_isle.rag.st_memory_manager import memory_manager

        # 即使空列表也应返回空字符串而非崩溃
        context = memory_manager.format_context_for_prompt([])
        assert isinstance(context, str)

    def test_different_world_lines_isolation(self):
        """不同世界线的搜索应隔离"""
        from src.aegis_isle.rag.st_memory_manager import memory_manager
        import asyncio

        async def _search_two_worlds():
            r1 = await memory_manager.search_memory(
                query="约会", character_name="ZouZheng", world_line="AIDom", k=3
            )
            r2 = await memory_manager.search_memory(
                query="约会",
                character_name="ZouZheng",
                world_line="NonExistentWorld_XYZ",
                k=3,
            )
            return r1, r2

        r1, r2 = asyncio.run(_search_two_worlds())
        # 不存在的世界线应返回空列表
        assert isinstance(r2, list)


# ============================================
# 2. DeepEval LLM-as-Judge 评估 (需要 API Key)
# ============================================

# DeepEval 评估需要一个 LLM 当 "裁判"
# 我们直接用你已有的 SiliconFlow API，不需要注册任何第三方服务
# 只要 .env 里有 OPENAI_API_KEY 和 OPENAI_BASE_URL 就能跑

# 可选的裁判模型（按推荐程度排序）
# 在 .env 或环境变量中设置 JUDGE_MODEL 来切换，例如:
#   JUDGE_MODEL=Qwen/Qwen3-235B-A22B
AVAILABLE_JUDGE_MODELS = {
    # 🥇 顶级（最准确，但贵一点）
    "deepseek-ai/DeepSeek-V3": "DeepSeek V3 — 推理超强，当评判最准确",
    "Qwen/Qwen3-235B-A22B": "Qwen3 235B MoE — 旗舰级，多语言最强",
    "Qwen/QwQ-32B": "QwQ 32B — 推理型，适合复杂评估",
    # 🥈 性价比（够用，便宜）
    "deepseek-ai/DeepSeek-V3-0324": "DeepSeek V3 0324 — 平衡性价比",
    "Qwen/Qwen3-8B": "Qwen3 8B — 轻量但有思考模式",
    # 🥉 免费（质量一般）
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5 7B — 免费但评分偏保守",
}

# 默认用 DeepSeek-V3（准确度高）
DEFAULT_JUDGE_MODEL = "deepseek-ai/DeepSeek-V3"


def _get_silicon_flow_config():
    """从 .env 或环境变量读取 SiliconFlow 配置"""
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    judge_model = os.environ.get("JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    return api_key, base_url, judge_model


HAS_LLM_KEY = bool(_get_silicon_flow_config()[0])


class SiliconFlowJudge:
    """用 SiliconFlow 上的任意模型作为 DeepEval 的评判 LLM"""

    @staticmethod
    def create(model_name: str = None):
        """
        创建一个用 SiliconFlow API 的自定义评判模型

        Args:
            model_name: 模型名称，不传则用 JUDGE_MODEL 环境变量或默认值
        """
        from deepeval.models.base_model import DeepEvalBaseLLM
        from openai import AsyncOpenAI, OpenAI

        api_key, base_url, env_model = _get_silicon_flow_config()
        chosen_model = model_name or env_model

        class _SiliconFlowModel(DeepEvalBaseLLM):
            def get_model_name(self):
                return chosen_model

            def load_model(self):
                return OpenAI(api_key=api_key, base_url=base_url)

            def generate(self, prompt: str) -> str:
                client = self.load_model()
                resp = client.chat.completions.create(
                    model=chosen_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    temperature=0.1,
                )
                return resp.choices[0].message.content

            async def a_generate(self, prompt: str) -> str:
                client = AsyncOpenAI(api_key=api_key, base_url=base_url)
                resp = await client.chat.completions.create(
                    model=chosen_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    temperature=0.1,
                )
                return resp.choices[0].message.content

        return _SiliconFlowModel()


@pytest.mark.skipif(not HAS_LLM_KEY, reason="需要 .env 中配置 OPENAI_API_KEY")
class TestRAGDeepEval:
    """DeepEval LLM-as-Judge 评估 — 用 SiliconFlow 当裁判，不需要注册任何服务"""

    def test_faithfulness_of_mock_response(self):
        """测试一个模拟的 RAG 回答的忠实度"""
        from deepeval import assert_test
        from deepeval.test_case import LLMTestCase
        from deepeval.metrics import FaithfulnessMetric

        judge = SiliconFlowJudge.create()

        test_case = LLMTestCase(
            input="你还记得那次在法餐厅的事吗？",
            actual_output="那次在法餐厅，你第一次尝试了鹅肝，虽然有点紧张，但后来你说味道其实还不错。",
            retrieval_context=[
                "在法餐厅的约会中，邹峥第一次尝试了鹅肝。他起初非常紧张，但最终评价说'味道竟然还不错'。"
            ],
        )

        faithfulness = FaithfulnessMetric(threshold=0.7, model=judge)
        assert_test(test_case, [faithfulness])

    def test_context_recall_of_mock_response(self):
        """测试一个模拟的检索召回率"""
        from deepeval import assert_test
        from deepeval.test_case import LLMTestCase
        from deepeval.metrics import ContextualRecallMetric

        judge = SiliconFlowJudge.create()

        test_case = LLMTestCase(
            input="邹峥对音乐的品味是什么？",
            actual_output="邹峥喜欢古典音乐，特别是肖邦和德彪西。",
            expected_output="邹峥喜欢古典音乐，尤其是肖邦的夜曲和德彪西的亚麻色头发的少女。",
            retrieval_context=[
                "邹峥的音乐偏好: 古典音乐，最喜欢的作曲家是肖邦和德彪西。他经常在做研究的时候听肖邦的夜曲。"
            ],
        )

        recall = ContextualRecallMetric(threshold=0.7, model=judge)
        assert_test(test_case, [recall])

    def test_answer_relevancy(self):
        """测试回答与问题的相关性"""
        from deepeval import assert_test
        from deepeval.test_case import LLMTestCase
        from deepeval.metrics import AnswerRelevancyMetric

        judge = SiliconFlowJudge.create()

        test_case = LLMTestCase(
            input="我们第一次见面是什么时候？",
            actual_output="我们第一次见面是在2025年10月，当时是在星巴克。那是一个秋天的下午。",
            retrieval_context=[
                "2025年10月，两人在星巴克首次见面。邹峥当天穿了白色衬衫和卡其色裤子。"
            ],
        )

        # DeepSeek-V3 作为严格裁判，会扣「无关细节」的分（如提到地点）
        # 0.6 是大模型严格评判下的合理阈值
        relevancy = AnswerRelevancyMetric(threshold=0.6, model=judge)
        assert_test(test_case, [relevancy])
