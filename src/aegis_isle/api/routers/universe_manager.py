"""
Universe Manager 路由
提供世界线管理所需的全部后端接口
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import os
import json
import glob
import logging
import time
from pathlib import Path

router = APIRouter()
logger = logging.getLogger(__name__)

# =====================
# 路径配置
# =====================
CHUNKS_DIR = Path("debug/chunks")
ALIASES_FILE = Path("data/universe_aliases.json")
FEEDBACK_FILE = Path("data/universe_feedback.jsonl")

def load_aliases() -> dict:
    if ALIASES_FILE.exists():
        try:
            return json.loads(ALIASES_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_aliases(aliases: dict):
    ALIASES_FILE.parent.mkdir(parents=True, exist_ok=True)
    ALIASES_FILE.write_text(json.dumps(aliases, ensure_ascii=False, indent=2), encoding="utf-8")


# =====================
# 接口一：列出某角色的所有宇宙
# =====================

@router.get("/universe/list")
async def list_universes(character: str):
    """
    列出某个角色下的所有可用宇宙，附带别名、episode数量、sub_chunk数量。
    """
    try:
        if not CHUNKS_DIR.exists():
            return JSONResponse({"universes": []})
        
        aliases = load_aliases()

        # 找出所有属于该角色的 universe_id（通过 episodes 文件名反推）
        pattern = str(CHUNKS_DIR / f"*_episodes.jsonl")
        all_episode_files = glob.glob(pattern)

        universes = []
        for ep_file in sorted(all_episode_files):
            basename = Path(ep_file).name  # e.g. 12岁_养父_..._episodes.jsonl
            universe_id = basename.replace("_episodes.jsonl", "")

            # 简单过滤：要求 universe_id 中包含角色名关键词（不精确但够用）
            # 用 episodes 里的 character_name 来精确过滤
            episodes = []
            try:
                with open(ep_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            episodes.append(json.loads(line))
            except Exception:
                pass

            if not episodes:
                continue

            char_name = episodes[0].get("character_name", "")
            if character.lower() not in char_name.lower():
                continue

            # 计算 sub_chunk 数量
            sub_chunk_file = str(CHUNKS_DIR / f"{universe_id}_sub_chunks.jsonl")
            sub_chunk_count = 0
            if os.path.exists(sub_chunk_file):
                try:
                    with open(sub_chunk_file, "r", encoding="utf-8") as f:
                        sub_chunk_count = sum(1 for line in f if line.strip())
                except Exception:
                    pass

            # 读取 universe_info.json（功能二的配置名称）
            info_file = Path("data/universes") / universe_id / "universe_info.json"
            universe_info = {}
            if info_file.exists():
                try:
                    universe_info = json.loads(info_file.read_text(encoding="utf-8"))
                except Exception:
                    pass

            universes.append({
                "universe_id": universe_id,
                "alias": aliases.get(universe_id, universe_id),
                "character_name": char_name,
                "episode_count": len(episodes),
                "chunk_count": sub_chunk_count,
                "serial_start": episodes[0].get("serial", ""),
                "time_start": episodes[0].get("time_range", ""),
                "universe_info": universe_info,
            })

        return JSONResponse({"universes": universes})

    except Exception as e:
        logger.error(f"[UniverseManager] list_universes 失败: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


# =====================
# 接口二：获取某宇宙的 episode 章节列表
# =====================

@router.get("/universe/episodes")
async def get_universe_episodes(universe_id: str):
    """
    返回某宇宙下的所有 episode（章节摘要），按 serial 排序。
    """
    try:
        ep_file = CHUNKS_DIR / f"{universe_id}_episodes.jsonl"
        if not ep_file.exists():
            return JSONResponse({"episodes": []})

        episodes = []
        with open(ep_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    episodes.append(json.loads(line))

        return JSONResponse({"episodes": episodes})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# =====================
# 接口三：获取某父 chunk 下的所有 sub_chunks
# =====================

@router.get("/universe/chunks")
async def get_sub_chunks(universe_id: str, parent_chunk_id: Optional[str] = None, limit: int = 100):
    """
    返回某宇宙的 sub_chunks。
    可通过 parent_chunk_id 过滤出某一章节的所有子切片。
    """
    try:
        sc_file = CHUNKS_DIR / f"{universe_id}_sub_chunks.jsonl"
        if not sc_file.exists():
            return JSONResponse({"chunks": []})

        chunks = []
        with open(sc_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunk = json.loads(line)
                    if parent_chunk_id and chunk.get("parent_chunk_id") != parent_chunk_id:
                        continue
                    chunks.append(chunk)
                    if len(chunks) >= limit:
                        break

        return JSONResponse({"chunks": chunks})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/universe/parent_chunks")
async def get_parent_chunks(universe_id: str):
    """
    返回某宇宙的所有 parent_chunks（包含 scene_id、user_msg、sub_chunk_ids 等字段）
    """
    try:
        pc_file = CHUNKS_DIR / f"{universe_id}_parent_chunks.jsonl"
        if not pc_file.exists():
            return JSONResponse({"parent_chunks": []})

        parents = []
        with open(pc_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    parents.append(json.loads(line))

        return JSONResponse({"parent_chunks": parents})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# =====================
# 接口四：重命名宇宙（设置别名）
# =====================

class RenameRequest(BaseModel):
    universe_id: str
    alias: str

@router.post("/universe/rename")
async def rename_universe(req: RenameRequest):
    try:
        aliases = load_aliases()
        aliases[req.universe_id] = req.alias
        save_aliases(aliases)
        return JSONResponse({"status": "ok", "universe_id": req.universe_id, "alias": req.alias})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# =====================
# 接口五：用 episode 摘要自动取名
# =====================

@router.post("/universe/auto_name")
async def auto_name_universe(universe_id: str):
    """
    读取该宇宙所有 episode 的 plot 字段，拼接后让 LLM 生成一个 10 字以内的宇宙名称。
    """
    try:
        ep_file = CHUNKS_DIR / f"{universe_id}_episodes.jsonl"
        if not ep_file.exists():
            return JSONResponse({"status": "error", "message": "没有 episodes 文件"})

        plots = []
        with open(ep_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    ep = json.loads(line)
                    if ep.get("plot"):
                        plots.append(ep["plot"][:100])  # 每章取前100字

        if not plots:
            return JSONResponse({"status": "error", "message": "无 plot 数据"})

        # 拼接所有 plot （最多取前 5 章避免 prompt 过长）
        combined_plots = "\n".join(plots[:5])

        from aegis_isle.core.config import settings
        from openai import AsyncOpenAI

        api_key = settings.openai_api_key
        base_url = settings.openai_base_url or "https://api.siliconflow.cn/v1"

        if not api_key:
            # Fallback：取第一章 serial 作为名称
            fallback_name = plots[0][:15] + "..."
            aliases = load_aliases()
            aliases[universe_id] = fallback_name
            save_aliases(aliases)
            return JSONResponse({"status": "ok", "alias": fallback_name, "method": "fallback"})

        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        response = await client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[
                {"role": "system", "content": "你是一个擅长为故事命名的文案大师。用户会给你一段剧情摘要，请你给这段故事起一个10字以内、富有文学感的中文标题，只输出标题本身，不加任何标点解释。"},
                {"role": "user", "content": f"以下是这段故事各章节的摘要：\n{combined_plots}\n\n请给这个故事起一个标题："}
            ],
            max_tokens=30,
            temperature=0.7
        )
        alias = response.choices[0].message.content.strip().strip("《》「」【】")

        aliases = load_aliases()
        aliases[universe_id] = alias
        save_aliases(aliases)

        return JSONResponse({"status": "ok", "alias": alias, "method": "llm"})

    except Exception as e:
        logger.error(f"[UniverseManager] auto_name 失败: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


# =====================
# 接口六：用户对 chunk 打分（功能五）
# =====================

class FeedbackRequest(BaseModel):
    chunk_id: str
    query: str
    score: int  # 1-5

@router.post("/universe/feedback")
async def save_chunk_feedback(req: FeedbackRequest):
    """保存用户对某个 chunk 的检索质量评分"""
    try:
        FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "chunk_id": req.chunk_id,
            "query": req.query,
            "score": req.score,
            "timestamp": time.time()
        }
        with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return JSONResponse({"status": "ok"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# =====================
# 接口七：带打分的语义搜索（功能五）
# =====================

@router.get("/universe/search")
async def search_with_score(
    query: str,
    character: str,
    k: int = 10,
    target_universes: Optional[str] = None,
    human_weight: float = 0.4
):
    """
    在指定角色的所有宇宙里进行语义搜索，返回 chunk 并附带历史平均打分。
    自动枚举该角色的所有 FAISS 索引文件，跨全宇宙并发检索。
    """
    try:
        from aegis_isle.rag.st_memory_manager import memory_manager
        import re

        # ── 1. 枚举该角色所有可用的 world_line ──
        vs_dir = memory_manager.vectorstore_dir  # e.g. "data/vectorstore/st_memory"
        safe_char = re.sub(r'[^\w\u4e00-\u9fff \-_]', '', character).strip()
        
        # 解析用户限定的目标宇宙 ID
        target_uids = set(target_universes.split(",")) if target_universes else set()
        index_pattern = os.path.join(vs_dir, f"{safe_char}*.index")
        index_files = glob.glob(index_pattern)

        if not index_files:
            return JSONResponse({"results": [], "message": f"未找到角色 {character} 的任何 FAISS 索引文件，路径: {index_pattern}"})

        # 从文件名反推 world_line
        world_lines = []
        for fp in index_files:
            basename = os.path.basename(fp).replace(".index", "")
            
            # 解析出 universe_id (恢复原始的 universe_id 作为匹配依据)
            # 由于索引文件名是 {safe_char}_{universe_id}.index
            wl = None
            if basename != safe_char:
                wl = basename[len(safe_char):].lstrip("_")
            
            # 如果有限定宇宙，过滤掉不相干的
            if target_uids:
                # 兼容：前端传来的 target_uid(即file_id) 可能带有 _20260306_211620 这种提取时间戳后缀
                # 而 FAISS 中的 wl 是没有这个后缀的。
                if not wl or not any(tuid == wl or tuid.startswith(wl + "_") for tuid in target_uids):
                    continue

            world_lines.append(wl)

        if not world_lines:
            return JSONResponse({"results": [], "message": "过滤后没有匹配的宇宙索引要搜索"})

        logger.info(f"[UniverseSearch] 角色={character} 找到 {len(world_lines)} 个索引，跨宇宙搜索...")

        # ── 2. 读取历史打分数据 ──
        feedback_scores: dict = {}
        if FEEDBACK_FILE.exists():
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        cid = rec.get("chunk_id", "")
                        if cid:
                            feedback_scores.setdefault(cid, []).append(rec.get("score", 3))
        avg_scores = {cid: sum(v)/len(v) for cid, v in feedback_scores.items()}

        # ── 3. 跨所有宇宙并发语义搜索 ──
        # 构造 comma-separated world_line 字符串让 search_memory 并发搜索
        wl_str = ",".join(w for w in world_lines if w is not None)
        # 分别搜索：有 world_line 的 + None（基础库）
        all_docs = []
        search_tasks = []

        import asyncio
        if None in world_lines:
            search_tasks.append(memory_manager.search_memory(query, character, world_line=None, k=k))
        if wl_str:
            search_tasks.append(memory_manager.search_memory(query, character, world_line=wl_str, k=k))

        gathered = await asyncio.gather(*search_tasks, return_exceptions=True)
        for result in gathered:
            if isinstance(result, list):
                all_docs.extend(result)

        if not all_docs:
            return JSONResponse({"results": [], "message": f"检索完成但无结果，共搜索 {len(world_lines)} 个宇宙索引"})

        # ── 4. Re-ranking：向量相似度 + 人类偏好 ──
        sim_weight = 1.0 - human_weight
        results = []
        seen = set()
        for doc in all_docs:
            text = doc.page_content
            if text in seen:
                continue
            seen.add(text)

            chunk_id = doc.metadata.get("chunk_id") or doc.metadata.get("source", "")
            human_score = avg_scores.get(chunk_id, 3.0)
            # FAISS L2 distance → 转换为 0~1 相似度（距离越小越好）
            raw_dist = doc.metadata.get("score", 1.0)
            similarity = max(0.0, 1.0 - float(raw_dist) / 2.0)
            final_score = similarity * sim_weight + (human_score / 5.0) * human_weight

            results.append({
                "chunk_id": chunk_id,
                "text": text,
                "metadata": doc.metadata,
                "similarity": round(similarity, 3),
                "human_avg_score": round(human_score, 2),
                "final_score": round(final_score, 3),
            })

        results.sort(key=lambda x: x["final_score"], reverse=True)
        return JSONResponse({"results": results[:k], "searched_universes": len(world_lines)})

    except Exception as e:
        logger.error(f"[UniverseManager] search_with_score 失败: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


# =====================
# 接口八：数据清洗诊断（让用户上传一段原始预设文本，自动识别风格并预览清洗结果）
# =====================

class DiagnoseRequest(BaseModel):
    raw_text: str             # 用户粘贴的原始 ST 聊天记录段落（AI 回复部分）
    user_msg: str = ""        # 用户的触发消息（可选，辅助判断）

def _detect_preset_style(text: str) -> dict:
    """
    自动探针：通过正则特征识别 ST 预设使用的排版风格。
    返回 {style_id, style_name, description, confidence}
    """
    import re
    scores = {}
    
    # 旁白+对话 - 星号加粗格式
    bold_matches = re.findall(r'\*{1,3}[^\*]{2,60}\*{1,3}', text)
    scores["bold_action"] = len(bold_matches) * 2

    # 日式引号对话
    jp_matches = re.findall(r'[「『].{2,60}[」』]', text)
    scores["japanese_quote"] = len(jp_matches) * 2

    # 括号内心戏
    paren_matches = re.findall(r'[（(][^\)）]{2,40}[)）]', text)
    scores["parenthesis_thought"] = len(paren_matches) * 2

    # 论坛/小剧场体（包含【】标记）
    forum_matches = re.findall(r'[【\[].{2,20}[】\]]', text)
    scores["forum_style"] = len(forum_matches) * 3

    # 代码块/Markdown 混入
    code_matches = re.findall(r'```[a-z]*', text)
    scores["markdown_noise"] = len(code_matches) * 5

    # HTML 标签混入
    html_matches = re.findall(r'<[a-zA-Z_]+[^>]*>', text)
    scores["html_noise"] = len(html_matches) * 4

    style_meta = {
        "bold_action":         {"name": "旁白+对话（*星号*加粗）", "desc": "以 *动作* 表示行为，「」或常规引号表示对话，常见于 中文RP"},
        "japanese_quote":      {"name": "日式引号纯对话体", "desc": "用「台词」表示对话，旁白为纯叙述"},
        "parenthesis_thought": {"name": "括号内心戏", "desc": "用（内心想法）表示人物心理，常见于女性向等 RP"},
        "forum_style":         {"name": "论坛体/小剧场", "desc": "含【小剧场】【标题】等 meta 标记，需整段过滤"},
        "markdown_noise":      {"name": "含 Markdown 代码块", "desc": "混有 ```html 或 ```mermaid 等渲染块，必须清除"},
        "html_noise":          {"name": "含 HTML 标签", "desc": "混有 <aurora_time> 等自定义标签"},
    }

    dominant = max(scores, key=scores.get) if any(scores.values()) else "plain"
    confidence = scores.get(dominant, 0)

    return {
        "detected_style": dominant,
        "style_name": style_meta.get(dominant, {}).get("name", "普通纯文本"),
        "description": style_meta.get(dominant, {}).get("desc", "无特殊格式标记"),
        "confidence_score": confidence,
        "all_scores": scores,
    }


def _apply_clean(text: str, style: str) -> str:
    """根据探测到的风格，应用对应的清洗策略"""
    import re
    
    # 通用：去除 HTML 注释、代码块、HTML 自定义标签
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    text = re.sub(r'```[\s\S]*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'</?(?:content|aurora_time|li|aurora)[^>]*>', '', text)

    if style == "forum_style":
        # 论坛体：整行过滤【】标记行（非正文）
        text = re.sub(r'^[【\[].*?[】\]].*$', '', text, flags=re.MULTILINE)

    elif style == "bold_action":
        # 星号旁白：保留星号内容但去掉星号本身（保留语义）
        text = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'[\1]', text)

    elif style == "parenthesis_thought":
        # 括号内心：改为 [内心:] 标记保留
        text = re.sub(r'[（(]([^)）]{2,60})[)）]', r'[内心:\1]', text)

    elif style == "markdown_noise":
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)

    # 通用：去末尾的 bgm / 状态栏装饰
    text = re.sub(r'(当前bgm:|⋯♡⋯|𐙚₊˚|☆₊⁺|𓋫 𓏴𓏴).*?(\n|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n\s*\n', '\n', text)
    return text.strip()


@router.post("/universe/diagnose_clean")
async def diagnose_clean(req: DiagnoseRequest):
    """
    接受用户粘贴的原始 ST 聊天片段，
    自动探测预设风格 → 预览清洗效果 → 给出优化建议。
    """
    try:
        raw = req.raw_text.strip()
        if not raw:
            return JSONResponse({"error": "raw_text 不能为空"}, status_code=400)

        style_info = _detect_preset_style(raw)
        cleaned = _apply_clean(raw, style_info["detected_style"])

        # 计算清洗率
        removed_chars = len(raw) - len(cleaned)
        clean_rate = round(removed_chars / max(len(raw), 1) * 100, 1)

        recommendations = []
        if style_info["confidence_score"] == 0:
            recommendations.append("未检测到明显格式标记，使用通用清洗策略即可。")
        if style_info["detected_style"] == "markdown_noise":
            recommendations.append("⚠️ 发现代码块（```html/mermaid）—— 强烈建议在导入前清除，否则会严重污染向量空间。")
        if "html_noise" in style_info["all_scores"] and style_info["all_scores"]["html_noise"] > 0:
            recommendations.append("⚠️ 发现 HTML 自定义标签（如 <aurora_time>）—— 已自动去除，不影响语义。")
        if style_info["detected_style"] == "forum_style":
            recommendations.append("ℹ️ 检测到论坛/小剧场格式 —— 【】整行标记已过滤，正文部分保留。")
        if clean_rate > 30:
            recommendations.append(f"🎯 清洗率 {clean_rate}% 偏高，建议检查原数据是否包含大量 UI 装饰符号。")

        return JSONResponse({
            "style": style_info,
            "cleaned_preview": cleaned[:800],   # 最多展示800字预览
            "original_length": len(raw),
            "cleaned_length": len(cleaned),
            "clean_rate_pct": clean_rate,
            "recommendations": recommendations,
        })

    except Exception as e:
        logger.error(f"[DiagnoseClean] 失败: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


# =====================
# 接口九：Query 语义增强（意图种子词注入）
# =====================

QUERY_SEED_PHRASES = {
    "回忆型": ["你还记得", "你还记不记得", "我说过", "我曾经说过", "我们之间", "上一次", "那一次", "当时"],
    "情感型": ["你对我的", "你喜不喜欢", "你有没有想过", "你最在乎的", "你的感受"],
    "事件型": ["发生了什么", "那件事", "我们做过", "一起经历的", "你告诉过我"],
    "地点型": ["在书房", "在旧申府", "在申都", "那个地方"],
    "时间型": ["第一次见面", "刚见面时", "很久以前", "那个下午", "刚认识时"],
}

@router.get("/universe/query_seed_phrases")
async def get_query_seed_phrases():
    """返回可用于 Query 增强的意图种子词列表（按类别分组）"""
    return JSONResponse({"categories": QUERY_SEED_PHRASES})
