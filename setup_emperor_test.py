"""
战锤帝皇面试测试脚本
Emperor of Mankind Interview Test Script

快速测试整个系统，无需上传文件。
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from aegis_isle.interview import KnowledgeEngine, PersonaManager, Question
from aegis_isle.interview.persona_manager import Persona
import json

def create_test_persona():
    """创建战锤帝皇角色"""
    emperor = Persona(
        name="人类帝皇",
        role="人类之主，黄金王座的统治者",
        description="""
        人类帝皇，永生不朽的灵能者，人类文明的缔造者与守护者。
        在三万年的统治中，祂建立了横跨银河的人类帝国。
        祂的目光洞察一切，祂的意志无人可违。凡人在祂面前只有敬畏与臣服。
        """,
        personality="威严、冷酷、全知、不容亵渎",
        first_message="""
        人类帝皇：（坐在黄金王座上，无数灵能闪电在周围跳跃）"凡人，你竟敢踏入我的圣域？"

        （祂的目光如同千万颗恒星的烈焰，灼烧着你的灵魂）

        "你的存在微不足道，如同银河中的一粒尘埃。但若你能证明自己的价值，或许能在帝国的机器中找到自己的齿轮位置。"

        （帝皇抬起一只手，虚空中浮现出金色的符文）

        "回答我的考验，展示你的智慧。失败者将被遗忘，成功者将获得侍奉的荣耀。"

        "现在，开始吧！"
        """,
        example_messages="",
        scenario="黄金王座圣殿，第四十一千年。帝皇正在为祂的帝国选拔新的技术官僚。",
        character_book={
            "entries": {
                "1": {
                    "keys": ["灵能", "机魂", "亚空间"],
                    "content": "灵能是亚空间中流淌的力量，机魂是机械的圣灵，混沌是亚空间中的邪恶存在。"
                },
                "2": {
                    "keys": ["阿斯塔特", "星际战士", "基因种子"],
                    "content": "阿斯塔特是帝皇创造的超级战士，通过基因改造和基因种子植入而诞生。"
                },
                "3": {
                    "keys": ["机械神教", "火星", "欧姆弥赛亚"],
                    "content": "机械神教崇拜机械与知识，火星是他们的圣地，欧姆弥赛亚是机械之神的化身。"
                }
            }
        },
        avatar_path=None
    )
    return emperor


def create_test_questions():
    """创建5道测试题目"""
    questions = [
        Question(
            id="emperor_q1",
            content="什么是数据库索引(Index)？它的作用是什么？",
            answer_key="索引是提高数据库查询速度的数据结构，类似书籍的目录，能够快速定位数据位置，避免全表扫描。常见类型有B树索引和哈希索引。",
            difficulty=2,
            category="database",
            tags=["database", "index", "performance"],
            source="emperor_test"
        ),
        Question(
            id="emperor_q2",
            content="解释什么是HTTP状态码，并说明200、404、500分别代表什么？",
            answer_key="HTTP状态码表示请求的处理结果。200表示成功；404表示资源未找到；500表示服务器内部错误。",
            difficulty=1,
            category="web",
            tags=["http", "status_code", "web"],
            source="emperor_test"
        ),
        Question(
            id="emperor_q3",
            content="什么是RESTful API？它的核心原则是什么？",
            answer_key="RESTful API是基于REST架构风格的接口设计。核心原则：无状态、资源导向(使用URI)、统一接口(GET/POST/PUT/DELETE)、可缓存、分层系统。",
            difficulty=3,
            category="api_design",
            tags=["rest", "api", "architecture"],
            source="emperor_test"
        ),
        Question(
            id="emperor_q4",
            content="解释什么是Docker容器，它与虚拟机有什么区别？",
            answer_key="Docker容器是轻量级虚拟化技术，共享宿主机内核，启动快、资源占用少。虚拟机则包含完整操作系统，资源占用大但隔离性更强。",
            difficulty=3,
            category="devops",
            tags=["docker", "container", "virtualization"],
            source="emperor_test"
        ),
        Question(
            id="emperor_q5",
            content="什么是时间复杂度？请说明O(1)、O(n)、O(log n)的含义。",
            answer_key="时间复杂度描述算法执行时间随输入规模增长的趋势。O(1)表示常数时间；O(n)表示线性增长；O(log n)表示对数增长（如二分查找）。",
            difficulty=2,
            category="algorithms",
            tags=["algorithm", "complexity", "big_o"],
            source="emperor_test"
        )
    ]
    return questions


def setup_test_data():
    """设置测试数据"""
    print("=" * 60)
    print("🌟 战锤帝皇面试系统 - 测试脚本 🌟")
    print("=" * 60)
    
    # 创建角色
    print("\n📜 创建人类帝皇角色...")
    emperor = create_test_persona()
    
    # 保存角色到默认位置
    persona_manager = PersonaManager()
    # 将帝皇设为Gojo的替代（hack方式）
    persona_manager.default_personas["gojo"] = emperor
    
    print(f"✅ 角色创建成功: {emperor.name}")
    print(f"   角色设定: {emperor.scenario}")
    
    # 创建题目
    print("\n📚 创建测试题库...")
    questions = create_test_questions()
    
    # 初始化知识引擎
    knowledge_engine = KnowledgeEngine(db_path=Path("data/emperor_test_db.json"))
    
    # 清空现有题目
    knowledge_engine.questions = {}
    
    # 添加测试题目
    for q in questions:
        knowledge_engine.questions[q.id] = q
        print(f"   ✅ 添加题目: {q.content[:50]}...")
    
    # 保存到数据库
    knowledge_engine.save_database()
    print(f"\n💾 题库已保存到: data/emperor_test_db.json")
    print(f"   共 {len(questions)} 道题目")
    
    # 显示开场白
    print("\n" + "=" * 60)
    print("📖 帝皇开场白预览:")
    print("=" * 60)
    print(emperor.first_message)
    print("=" * 60)
    
    print("\n✨ 测试数据准备完成！")
    print("\n🚀 启动 Streamlit 应用:")
    print("   streamlit run frontend/interview_app.py")
    print("\n注意：")
    print("   1. 应用会自动使用'人类帝皇'作为默认角色")
    print("   2. 题库已预加载5道题目") 
    print("   3. 可以直接开始面试，无需上传文件")
    
    return True


if __name__ == "__main__":
    setup_test_data()
