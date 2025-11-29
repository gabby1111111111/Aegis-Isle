"""
战锤帝皇面试测试数据
Emperor Test Data - Direct JSON Generation
"""

import json
from pathlib import Path
from datetime import datetime

def create_emperor_test_data():
    """创建帝皇测试数据（纯JSON，无依赖）"""
    
    # 创建 data 目录
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # 5道测试题目
    questions = {
        "emperor_q1": {
            "id": "emperor_q1",
            "content": "什么是数据库索引(Index)？它的作用是什么？",
            "answer_key": "索引是提高数据库查询速度的数据结构，类似书籍的目录，能够快速定位数据位置，避免全表扫描。常见类型有B树索引和哈希索引。",
            "difficulty": 2,
            "review_box": 0,
            "next_review": datetime.utcnow().isoformat(),
            "created_at": datetime.utcnow().isoformat(),
            "category": "database",
            "tags": ["database", "index", "performance"],
            "source": "emperor_test",
            "attempts": 0,
            "correct_answers": 0
        },
        "emperor_q2": {
            "id": "emperor_q2",
            "content": "解释什么是HTTP状态码，并说明200、404、500分别代表什么？",
            "answer_key": "HTTP状态码表示请求的处理结果。200表示成功；404表示资源未找到；500表示服务器内部错误。",
            "difficulty": 1,
            "review_box": 0,
            "next_review": datetime.utcnow().isoformat(),
            "created_at": datetime.utcnow().isoformat(),
            "category": "web",
            "tags": ["http", "status_code", "web"],
            "source": "emperor_test",
            "attempts": 0,
            "correct_answers": 0
        },
        "emperor_q3": {
            "id": "emperor_q3",
            "content": "什么是RESTful API？它的核心原则是什么？",
            "answer_key": "RESTful API是基于REST架构风格的接口设计。核心原则：无状态、资源导向(使用URI)、统一接口(GET/POST/PUT/DELETE)、可缓存、分层系统。",
            "difficulty": 3,
            "review_box": 0,
            "next_review": datetime.utcnow().isoformat(),
            "created_at": datetime.utcnow().isoformat(),
            "category": "api_design",
            "tags": ["rest", "api", "architecture"],
            "source": "emperor_test",
            "attempts": 0,
            "correct_answers": 0
        },
        "emperor_q4": {
            "id": "emperor_q4",
            "content": "解释什么是Docker容器，它与虚拟机有什么区别？",
            "answer_key": "Docker容器是轻量级虚拟化技术，共享宿主机内核，启动快、资源占用少。虚拟机则包含完整操作系统，资源占用大但隔离性更强。",
            "difficulty": 3,
            "review_box": 0,
            "next_review": datetime.utcnow().isoformat(),
            "created_at": datetime.utcnow().isoformat(),
            "category": "devops",
            "tags": ["docker", "container", "virtualization"],
            "source": "emperor_test",
            "attempts": 0,
            "correct_answers": 0
        },
        "emperor_q5": {
            "id": "emperor_q5",
            "content": "什么是时间复杂度？请说明O(1)、O(n)、O(log n)的含义。",
            "answer_key": "时间复杂度描述算法执行时间随输入规模增长的趋势。O(1)表示常数时间；O(n)表示线性增长；O(log n)表示对数增长（如二分查找）。",
            "difficulty": 2,
            "review_box": 0,
            "next_review": datetime.utcnow().isoformat(),
            "created_at": datetime.utcnow().isoformat(),
            "category": "algorithms",
            "tags": ["algorithm", "complexity", "big_o"],
            "source": "emperor_test",
            "attempts": 0,
            "correct_answers": 0
        }
    }
    
    # 帝皇角色卡（SillyTavern格式）
    emperor_card = {
        "name": "人类帝皇",
        "description": "人类帝皇，永生不朽的灵能者，人类文明的缔造者与守护者。在三万年的统治中，祂建立了横跨银河的人类帝国。祂的目光洞察一切，祂的意志无人可违。凡人在祂面前只有敬畏与臣服。",
        "personality": "威严、冷酷、全知、不容亵渎",
        "scenario": "黄金王座圣殿，第四十一千年。帝皇正在为祂的帝国选拔新的技术官僚。你站在无数骷髅和机械装置包围的黄金王座前，等待审判。",
        "first_mes": "人类帝皇：（坐在黄金王座上，无数灵能闪电在周围跳跃）\"凡人，你竟敢踏入我的圣域？\"\n\n（祂的目光如同千万颗恒星的烈焰，灼烧着你的灵魂）\n\n\"你的存在微不足道，如同银河中的一粒尘埃。但若你能证明自己的价值，或许能在帝国的机器中找到自己的齿轮位置。\"\n\n（帝皇抬起一只手，虚空中浮现出金色的符文）\n\n\"回答我的考验，展示你的智慧。失败者将被遗忘，成功者将获得侍奉的荣耀。\"\n\n\"现在，开始吧！\"",
        "mes_example": "",
        "creator_notes": "战锤40K宇宙的人类帝皇，适用于技术面试场景",
        "system_prompt": "你是人类帝皇，全知全能的永生者。你的话语充满威严与力量。",
        "post_history_instructions": "",
        "tags": ["warhammer", "emperor", "40k", "interview"],
        "creator": "Aegis Isle Test",
        "character_version": "1.0",
        "alternate_greetings": [],
        "extensions": {
            "world": "战锤40K",
            "depth_prompt": {
                "prompt": "",
                "depth": 4
            }
        },
        "character_book": {
            "entries": {
                "0": {
                    "id": 0,
                    "keys": ["灵能", "机魂", "亚空间", "混沌"],
                    "content": "灵能是亚空间中流淌的超自然力量，机魂是机械的圣灵，混沌是亚空间中的四大邪神(恐虐、奸奇、纳垢、色孽)。帝国的灵能者受到严格管制。",
                    "enabled": True,
                    "insertion_order": 100,
                    "case_sensitive": False,
                    "name": "基础设定",
                    "priority": 10,
                    "comment": ""
                },
                "1": {
                    "id": 1,
                    "keys": ["阿斯塔特", "星际战士", "基因种子"],
                    "content": "阿斯塔特是帝皇创造的超级战士，通过基因改造和基因种子植入而诞生。他们是人类最强大的战士，每一位都能以一敌百。",
                    "enabled": True,
                    "insertion_order": 100,
                    "case_sensitive": False,
                    "name": "阿斯塔特",
                    "priority": 9,
                    "comment": ""
                },
                "2": {
                    "id": 2,
                    "keys": ["机械神教", "火星", "欧姆弥赛亚", "机油佬"],
                    "content": "机械神教崇拜机械与知识，火星是他们的圣地。技术祭司们相信知识就是力量，机械的圣灵必须被安抚。他们将技术视为神圣仪式。",
                    "enabled": True,
                    "insertion_order": 100,
                    "case_sensitive": False,
                    "name": "机械神教",
                    "priority": 8,
                    "comment": ""
                }
            }
        }
    }
    
    # 保存题库
    db_path = data_dir / "emperor_test_db.json"
    with open(db_path, 'w', encoding='utf-8') as f:
        json.dump({"questions": questions}, f, ensure_ascii=False, indent=2)
    
    # 保存角色卡
    card_path = data_dir / "emperor_card.json"
    with open(card_path, 'w', encoding='utf-8') as f:
        json.dump(emperor_card, f, ensure_ascii=False, indent=2)
    
    return db_path, card_path


if __name__ == "__main__":
    print("=" * 70)
    print("🌟 战锤帝皇面试系统 - 测试数据生成器 🌟")
    print("=" * 70)
    
    db_path, card_path = create_emperor_test_data()
    
    print(f"\n✅ 生成完成！")
    print(f"\n📁 生成的文件：")
    print(f"   题库: {db_path}")
    print(f"   角色卡: {card_path}")
    
    print(f"\n📋 题目列表：")
    print("   1. 数据库索引")
    print("   2. HTTP状态码")
    print("   3. RESTful API")
    print("   4. Docker容器")
    print("   5. 时间复杂度")
    
    print(f"\n👑 角色: 人类帝皇")
    print("   开场：黄金王座圣殿")
    
    print(f"\n🚀 使用方法：")
    print("   1. 在 Streamlit 侧边栏上传 emperor_card.json")
    print("   2. 点击'召唤角色'")
    print("   3. 在配置页面点击'载入测试题库'")
    print("      （或通过代码加载 emperor_test_db.json）")
    print("   4. 开始面试！")
    
    print("\n" + "=" * 70)
    print("帝皇的开场白预览：")
    print("=" * 70)
    print("""
人类帝皇：（坐在黄金王座上，无数灵能闪电在周围跳跃）"凡人，你竟敢踏入我的圣域？"

（祂的目光如同千万颗恒星的烈焰，灼烧着你的灵魂）

"你的存在微不足道，如同银河中的一粒尘埃。但若你能证明自己的价值，
或许能在帝国的机器中找到自己的齿轮位置。"

（帝皇抬起一只手，虚空中浮现出金色的符文）

"回答我的考验，展示你的智慧。失败者将被遗忘，成功者将获得侍奉的荣耀。"

"现在，开始吧！"
    """)
    print("=" * 70)
