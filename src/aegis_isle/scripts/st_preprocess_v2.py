"""
ST聊天数据预处理脚本 v2
支持JSONL格式（ST原生导出）
输出：sub_chunks / parent_chunks / graph_nodes / graph_edges / episodes
"""

import re
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
import os

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────


@dataclass
class SceneMeta:
    date: str = ""
    time: str = ""
    location: str = ""
    weather: str = ""
    environment: str = ""
    costumes: dict = field(default_factory=dict)
    bgm: str = ""


@dataclass
class SubChunk:
    sub_chunk_id: str
    parent_chunk_id: str
    scene_id: str
    universe_id: str
    character_name: str
    text: str
    dh_index: int


@dataclass
class ParentChunk:
    parent_chunk_id: str
    scene_id: str
    universe_id: str
    character_name: str
    scene_meta: dict
    user_msg: str
    full_ai_text: str
    sub_chunk_ids: list


@dataclass
class GraphNode:
    node_id: str
    node_type: str
    name: str
    universe_id: str
    attributes: dict


@dataclass
class GraphEdge:
    edge_id: str
    source: str
    target: str
    relation: str
    scene_id: str
    universe_id: str
    sentiment: str = ""


@dataclass
class Episode:
    episode_id: str
    universe_id: str
    character_name: str
    serial: str
    time_range: str
    scene: str
    plot: str
    seeds: list


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────


def strip_html_tags(text):
    return re.sub(r"<[^>]+>", "", text).strip()


def remove_comment_blocks(text):
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)


def remove_side_characters(text):
    return re.sub(r"^>.*$", "", text, flags=re.MULTILINE)


def clean_blank_lines(text):
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def detect_universe(first_user_msg: str, filename: str) -> str:
    """
    从第一条用户消息+文件名推断宇宙ID
    规则：取文件名去掉日期部分，再截取前20字作为宇宙标识
    """
    # 文件名格式：宇宙名称_-_日期.jsonl
    stem = Path(filename).stem
    parts = stem.split("_-_")
    universe_name = parts[0].strip() if parts else stem
    # 简化为合法ID
    universe_id = re.sub(r"[^\w\u4e00-\u9fff]", "_", universe_name)[:30]
    return universe_id


# ─────────────────────────────────────────────
# 解析器（复用v1逻辑）
# ─────────────────────────────────────────────


def parse_aurora_time(raw):
    m = re.search(r"<aurora_time>(.*?)</aurora_time>", raw, re.DOTALL)
    if not m:
        return SceneMeta()
    block = m.group(1)
    meta = SceneMeta()
    t = re.search(r"时间：(.+?)(?:\n|$)", block)
    if t:
        time_str = t.group(1).strip()
        date_m = re.search(r"(\d{4}年\d{1,2}月\d{1,2}日)", time_str)
        time_m = re.search(r"(\d{1,2}:\d{2})", time_str)
        if date_m:
            meta.date = date_m.group(1)
        if time_m:
            meta.time = time_m.group(1)
    loc = re.search(r"地点：(.+?)(?:\n|$)", block)
    if loc:
        meta.location = loc.group(1).strip()
    weather = re.search(r"天气：(.+?)(?:\n|$)", block)
    if weather:
        meta.weather = re.sub(
            r"[^\w\s，。、：\u4e00-\u9fff]", "", weather.group(1)
        ).strip()
    env = re.search(r"环境：(.+?)(?:\n|$)", block)
    if env:
        meta.environment = env.group(1).strip()
    for c in re.findall(r"<li>(.+?)</li>", block):
        if "：" in c:
            name, outfit = c.split("：", 1)
            meta.costumes[name.strip()] = outfit.strip()
    bgm_m = re.search(r"<bgm[^>]*>(.*?)</bgm>", raw, re.DOTALL)
    if bgm_m:
        meta.bgm = strip_html_tags(bgm_m.group(1)).strip()
    return meta


def extract_final_paragraphs(content_block):
    text = re.sub(r"<!-- 创作反思.*?-->", "", content_block, flags=re.DOTALL)
    text = re.sub(r"<!-- 禁词.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"<!-- \(\d.*?-->", "", text, flags=re.DOTALL)
    parts = re.split(r"-->", text)
    paragraphs = []
    for part in parts:
        part = re.sub(r"<!--.*", "", part, flags=re.DOTALL)
        part = remove_side_characters(part)
        part = strip_html_tags(part)
        part = clean_blank_lines(part)
        if len(part.strip()) > 20:
            paragraphs.append(part.strip())
    return paragraphs


def _is_code_block(text):
    """检测是否为 CSS/HTML/JS 代码块（对 RAG 语义检索无意义）"""
    # CSS 特征：多个 { property: value } 模式
    css_hits = len(
        re.findall(
            r"\{[^}]*(?:background-color|font-size|margin|padding|display|border|color|width|height|text-align|font-family):",
            text,
        )
    )
    if css_hits >= 3:
        return True
    # HTML 标签占比过高
    html_tags = len(
        re.findall(
            r"<(?:div|span|style|script|table|html|head|body|link|meta)\b", text, re.I
        )
    )
    if html_tags >= 3:
        return True
    # @keyframes / @import 动画/导入
    if re.search(r"@(?:keyframes|import|media)\s", text):
        return True
    # ```html / ```css / ```mermaid 代码块（整块都是代码）
    if re.match(r"\s*```(?:html|css|js|mermaid)", text):
        return True
    return False


def refine_paragraphs(paragraphs, max_len=500, min_len=20):
    """
    二次切分：对超过 max_len 的段落按 \n\n 自然段落拆分。
    如果拆完仍有 > max_len 的碎片，再按中文句号等标点追加切分。
    同时过滤掉 CSS/HTML/JS 代码块。
    """
    result = []
    for para in paragraphs:
        # 过滤代码块
        if _is_code_block(para):
            continue

        if len(para) <= max_len:
            result.append(para)
            continue

        # 第一刀：按 \n\n 切
        sub_parts = [p.strip() for p in para.split("\n\n") if len(p.strip()) > min_len]

        if len(sub_parts) <= 1:
            # \n\n 切不动，尝试按单 \n 切
            sub_parts = [
                p.strip() for p in para.split("\n") if len(p.strip()) > min_len
            ]

        if len(sub_parts) <= 1:
            # 单 \n 也切不动，按中文句末标点切
            sentences = re.split(r"(?<=[。！？…」』])", para)
            # 合并短句到不超过 max_len
            buf = ""
            for s in sentences:
                if len(buf) + len(s) > max_len and buf:
                    result.append(buf.strip())
                    buf = s
                else:
                    buf += s
            if buf.strip() and len(buf.strip()) > min_len:
                result.append(buf.strip())
            continue

        # \n\n 或 \n 切成功了，递归检查每个子段是否还需要切
        for sp in sub_parts:
            if _is_code_block(sp):
                continue
            if len(sp) > max_len:
                result.extend(refine_paragraphs([sp], max_len, min_len))
            else:
                result.append(sp)

    return result


def parse_meow_fm(raw, universe_id, character_name, chunk_index):
    m = re.search(r"<meow_FM>(.*?)</meow_FM>", raw, re.DOTALL)
    if not m:
        return None
    block = remove_comment_blocks(m.group(1))
    ep = Episode(
        episode_id=f"ep_{universe_id}_{chunk_index:03d}",
        universe_id=universe_id,
        character_name=character_name,
        serial="",
        time_range="",
        scene="",
        plot="",
        seeds=[],
    )
    serial = re.search(r"serial:(.*?)(?:\n|$)", block)
    if serial:
        ep.serial = serial.group(1).strip()
    time_r = re.search(r"time:(.*?)(?:\n|$)", block)
    if time_r:
        ep.time_range = time_r.group(1).strip()
    scene = re.search(r"scene:(.*?)(?=plot:|$)", block, re.DOTALL)
    if scene:
        ep.scene = strip_html_tags(scene.group(1)).strip()
    plot = re.search(r"plot:(.*?)(?=seeds:|$)", block, re.DOTALL)
    if plot:
        ep.plot = plot.group(1).strip()
    seeds_block = re.search(r"seeds:(.*?)$", block, re.DOTALL)
    if seeds_block:
        seed_items = re.findall(r"<p[^>]*>(.*?)</p>", seeds_block.group(1), re.DOTALL)
        ep.seeds = [
            strip_html_tags(s).strip() for s in seed_items if len(s.strip()) > 5
        ]
    return ep


def parse_table_edit(raw, universe_id, scene_id):
    m = re.search(r"<tableEdit>(.*?)</tableEdit>", raw, re.DOTALL)
    if not m:
        return [], []
    block = m.group(1)
    nodes, edges = [], []
    insert_rows = re.findall(r"insertRow\((\d+),\s*\{(.*?)\}\)", block, re.DOTALL)
    for table_id_str, fields_str in insert_rows:
        table_id = int(table_id_str)
        fields = {}
        for fm in re.finditer(r'(\d+):\s*"([^"]*)"', fields_str):
            fields[int(fm.group(1))] = fm.group(2).strip()
        for fm in re.finditer(r"(\d+):\s*(\d+)(?=\s*[,}])", fields_str):
            idx = int(fm.group(1))
            if idx not in fields:
                fields[idx] = fm.group(2).strip()

        if table_id == 1:
            name = fields.get(0, "")
            if name:
                nodes.append(
                    GraphNode(
                        node_id=f"{universe_id}_char_{name}",
                        node_type="character",
                        name=name,
                        universe_id=universe_id,
                        attributes={
                            "appearance": fields.get(1, ""),
                            "personality": fields.get(2, ""),
                            "occupation": fields.get(3, ""),
                            "hobbies": fields.get(4, ""),
                            "preferences": fields.get(5, ""),
                            "residence": fields.get(6, ""),
                            "tags": fields.get(7, ""),
                        },
                    )
                )
        elif table_id == 2:
            char = fields.get(0, "")
            if char:
                edges.append(
                    GraphEdge(
                        edge_id=f"{universe_id}_rel_{char}_gabby",
                        source=f"{universe_id}_char_{char}",
                        target=f"{universe_id}_char_gabby",
                        relation=fields.get(1, ""),
                        scene_id=scene_id,
                        universe_id=universe_id,
                        sentiment=f"{fields.get(2, '')} 好感度:{fields.get(3, '0')}%",
                    )
                )
        elif table_id == 4:
            participants = fields.get(0, "")
            description = fields.get(1, "")
            date = fields.get(2, "")
            location = fields.get(3, "")
            emotion = fields.get(4, "")
            event_id = f"{universe_id}_event_{date}_{location}".replace(" ", "_")
            nodes.append(
                GraphNode(
                    node_id=event_id,
                    node_type="event",
                    name=description[:40] + "..."
                    if len(description) > 40
                    else description,
                    universe_id=universe_id,
                    attributes={
                        "full_description": description,
                        "date": date,
                        "location": location,
                        "emotion": emotion,
                        "participants": participants,
                    },
                )
            )
            for p in participants.split("/"):
                p = p.strip()
                if p:
                    edges.append(
                        GraphEdge(
                            edge_id=f"{universe_id}_participated_{p}_{event_id}",
                            source=f"{universe_id}_char_{p}",
                            target=event_id,
                            relation="参与了",
                            scene_id=scene_id,
                            universe_id=universe_id,
                            sentiment=emotion,
                        )
                    )
    return nodes, edges


# ─────────────────────────────────────────────
# 核心：解析单个JSONL文件
# ─────────────────────────────────────────────


def process_jsonl_file(filepath: str) -> dict:
    filepath = Path(filepath)
    lines = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    lines.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"跳过无效 JSON 行 (行号 {len(lines) + 1}): {e}")

    if not lines:
        return None

    # 第1行：文件元数据
    meta_row = lines[0]
    character_name = meta_row.get("character_name", "未知角色")
    create_date = meta_row.get("create_date", "")

    # 消息行
    messages = lines[1:]
    user_msgs = [m for m in messages if m.get("is_user", False)]
    ai_msgs = [
        m
        for m in messages
        if not m.get("is_user", False) and not m.get("is_system", False)
    ]

    # 宇宙ID：从文件名推断
    first_user_text = user_msgs[0]["mes"] if user_msgs else ""
    universe_id = detect_universe(first_user_text, filepath.name)

    # ── 格式检测：只保留有 aurora_time + content 标签的新格式消息 ──
    valid_ai_msgs = [
        m
        for m in ai_msgs
        if "<aurora_time>" in m.get("mes", "") and "<content>" in m.get("mes", "")
    ]
    skip_count = len(ai_msgs) - len(valid_ai_msgs)

    print(f"\n📂 {filepath.name}")
    print(f"   角色: {character_name} | 宇宙: {universe_id}")
    print(
        f"   AI消息: {len(ai_msgs)} | 新格式: {len(valid_ai_msgs)} | 跳过早期格式: {skip_count}"
    )

    if len(valid_ai_msgs) == 0:
        print("   ⚠️  全部均为早期格式，跳过此文件")
        return {
            "universe_id": universe_id,
            "character_name": character_name,
            "create_date": create_date,
            "skipped": True,
            "skip_reason": "无新格式消息",
            "sub_chunks": [],
            "parent_chunks": [],
            "graph_nodes": [],
            "graph_edges": [],
            "episodes": [],
            "stats": {"ai_messages": 0, "sub_chunks": 0, "scenes": 0},
        }

    ai_msgs = valid_ai_msgs  # 只处理新格式消息

    all_sub_chunks, all_parent_chunks = [], []
    all_graph_nodes, all_graph_edges, all_episodes = [], [], []
    scene_counter = {}

    # 构建用户消息时间索引，用于与 AI 消息精确配对
    user_msgs_sorted = sorted(
        [m for m in user_msgs if m.get("send_date")],
        key=lambda m: m.get("send_date", 0),
    )

    def find_user_msg_before(ai_send_date):
        """找到 AI 消息之前最近的一条用户消息"""
        best = None
        for um in user_msgs_sorted:
            if um.get("send_date", 0) <= ai_send_date:
                best = um
            else:
                break
        return best.get("mes", "") if best else ""

    for idx, ai_msg in enumerate(ai_msgs, 1):
        mes = ai_msg.get("mes", "")

        # 基于 send_date 精确配对用户消息，回退到索引配对
        ai_send_date = ai_msg.get("send_date", 0)
        if ai_send_date and user_msgs_sorted:
            user_msg_text = find_user_msg_before(ai_send_date)
        else:
            user_msg_text = (
                user_msgs[idx - 1]["mes"] if idx - 1 < len(user_msgs) else ""
            )

        # 场景metadata
        scene_meta = parse_aurora_time(mes)
        scene_key = f"{scene_meta.date}_{scene_meta.location}"
        if scene_key not in scene_counter:
            scene_counter[scene_key] = (
                f"{universe_id}_scene_{len(scene_counter) + 1:03d}"
            )
        scene_id = scene_counter[scene_key]
        chunk_id = f"{universe_id}_chunk_{idx:03d}"

        # 子chunk
        content_m = re.search(r"<content>(.*?)</content>", mes, re.DOTALL)
        sub_chunk_ids = []
        full_ai_text = ""

        if content_m:
            paragraphs = extract_final_paragraphs(content_m.group(1))
            paragraphs = refine_paragraphs(paragraphs, max_len=500, min_len=20)
            for i, para in enumerate(paragraphs):
                sub_id = f"{chunk_id}_p{i + 1:03d}"
                all_sub_chunks.append(
                    asdict(
                        SubChunk(
                            sub_chunk_id=sub_id,
                            parent_chunk_id=chunk_id,
                            scene_id=scene_id,
                            universe_id=universe_id,
                            character_name=character_name,
                            text=para,
                            dh_index=i + 1,
                        )
                    )
                )
                sub_chunk_ids.append(sub_id)
            full_ai_text = "\n\n".join(paragraphs)

        all_parent_chunks.append(
            asdict(
                ParentChunk(
                    parent_chunk_id=chunk_id,
                    scene_id=scene_id,
                    universe_id=universe_id,
                    character_name=character_name,
                    scene_meta=asdict(scene_meta),
                    user_msg=user_msg_text,
                    full_ai_text=full_ai_text,
                    sub_chunk_ids=sub_chunk_ids,
                )
            )
        )

        # Episode
        ep = parse_meow_fm(mes, universe_id, character_name, idx)
        if ep:
            all_episodes.append(asdict(ep))

        # Graph
        nodes, edges = parse_table_edit(mes, universe_id, scene_id)
        all_graph_nodes.extend([asdict(n) for n in nodes])
        all_graph_edges.extend([asdict(e) for e in edges])

    # 去重节点
    seen = {}
    for node in all_graph_nodes:
        nid = node["node_id"]
        if nid not in seen:
            seen[nid] = node
        else:
            for k, v in node["attributes"].items():
                if v and not seen[nid]["attributes"].get(k):
                    seen[nid]["attributes"][k] = v

    result_data = {
        "universe_id": universe_id,
        "character_name": character_name,
        "create_date": create_date,
        "sub_chunks": all_sub_chunks,
        "parent_chunks": all_parent_chunks,
        "graph_nodes": list(seen.values()),
        "graph_edges": all_graph_edges,
        "episodes": all_episodes,
        "stats": {
            "ai_messages": len(ai_msgs),
            "sub_chunks": len(all_sub_chunks),
            "scenes": len(scene_counter),
        },
    }

    # Debug Save
    if os.environ.get("DEBUG_SAVE", "").lower() == "true":
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_dir = Path("debug/chunks")
        debug_dir.mkdir(parents=True, exist_ok=True)

        safe_universe_id = re.sub(r"[^\w\u4e00-\u9fff]", "_", universe_id)
        file_prefix = f"{safe_universe_id}_{timestamp}"

        for key in [
            "sub_chunks",
            "parent_chunks",
            "graph_nodes",
            "graph_edges",
            "episodes",
        ]:
            debug_file = debug_dir / f"{file_prefix}_{key}.jsonl"
            with open(debug_file, "w", encoding="utf-8") as f:
                for item in result_data[key]:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"   [DEBUG_SAVE] 已写入调试文件至 {debug_dir}")

    return result_data


# ─────────────────────────────────────────────
# 批量处理入口
# ─────────────────────────────────────────────


def process_folder(folder: str, output_dir: str = "./output"):
    folder = Path(folder)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    jsonl_files = list(folder.glob("*.jsonl"))
    print(f"找到 {len(jsonl_files)} 个JSONL文件")

    all_data = {
        "sub_chunks": [],
        "parent_chunks": [],
        "graph_nodes": [],
        "graph_edges": [],
        "episodes": [],
    }
    summary = []

    for fpath in sorted(jsonl_files):
        result = process_jsonl_file(str(fpath))
        if not result:
            continue
        for key in all_data:
            all_data[key].extend(result[key])
        summary.append(
            {
                "file": fpath.name,
                "universe_id": result["universe_id"],
                "character_name": result["character_name"],
                "create_date": result["create_date"],
                **result["stats"],
            }
        )

    # 全局去重graph_nodes
    seen = {}
    for node in all_data["graph_nodes"]:
        nid = node["node_id"]
        if nid not in seen:
            seen[nid] = node
        else:
            for k, v in node["attributes"].items():
                if v and not seen[nid]["attributes"].get(k):
                    seen[nid]["attributes"][k] = v
    all_data["graph_nodes"] = list(seen.values())

    # 写文件
    for name, data in all_data.items():
        with open(output_path / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 写汇总表
    with open(output_path / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print("📊 处理完成")
    print(f"{'=' * 60}")
    print(f"{'文件':<35} {'宇宙ID':<20} {'AI消息':>6} {'子chunk':>7} {'场景':>4}")
    print("-" * 60)
    for s in summary:
        print(
            f"{s['file'][:33]:<35} {s['universe_id'][:18]:<20} {s['ai_messages']:>6} {s['sub_chunks']:>7} {s['scenes']:>4}"
        )
    print("-" * 60)
    print(
        f"总计: sub_chunks={len(all_data['sub_chunks'])} | graph_nodes={len(all_data['graph_nodes'])} | episodes={len(all_data['episodes'])}"
    )


# ─────────────────────────────────────────────
# 单文件测试
# ─────────────────────────────────────────────


def test_single(filepath: str):
    result = process_jsonl_file(filepath)
    if not result:
        print("解析失败")
        return

    print(f"\n宇宙ID: {result['universe_id']}")
    print(f"角色: {result['character_name']}")
    print("\n--- 第1个parent_chunk ---")
    pc = result["parent_chunks"][0]
    print(f"user_msg: {pc['user_msg'][:80]}")
    print(f"scene_meta: {json.dumps(pc['scene_meta'], ensure_ascii=False)}")
    print(f"sub_chunk数: {len(pc['sub_chunk_ids'])}")

    print("\n--- 前3个sub_chunks ---")
    for sc in result["sub_chunks"][:3]:
        print(f"\n[{sc['sub_chunk_id']}] universe={sc['universe_id']}")
        print(sc["text"][:150] + "...")

    print("\n--- Graph节点 ---")
    for node in result["graph_nodes"]:
        print(f"[{node['node_type']}] {node['name']} | universe={node['universe_id']}")
        for k, v in node["attributes"].items():
            if v:
                print(f"   {k}: {v}")

    print("\n--- Graph边 ---")
    for e in result["graph_edges"][:5]:
        print(f"  {e['source']} --{e['relation']}--> {e['target']} | {e['sentiment']}")

    print("\n--- Episodes ---")
    for ep in result["episodes"][:2]:
        print(f"  [{ep['serial']}] {ep['plot'][:80]}...")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法:")
        print("  单文件测试: python st_preprocess_v2.py test file.jsonl")
        print("  批量处理:   python st_preprocess_v2.py batch ./聊天记录/ ./output")
        sys.exit(1)
    mode = sys.argv[1]
    if mode == "test":
        test_single(sys.argv[2])
    elif mode == "batch":
        process_folder(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "./output")
