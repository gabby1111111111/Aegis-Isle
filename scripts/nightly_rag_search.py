import os
import glob
import json
import random
import time
import asyncio
import httpx
from itertools import combinations
import re

# We test these window sizes
WINDOW_SIZES = [100, 200, 300, 500, 800]
API_KEY = os.getenv("OPENAI_API_KEY", "")  # Will be loaded or fetched
API_URL = "https://api.siliconflow.cn/v1/chat/completions"


def centered_extract(full_text, sub_chunk_text, window_size):
    if sub_chunk_text and sub_chunk_text in full_text:
        hit_pos = full_text.index(sub_chunk_text)
        center = hit_pos + len(sub_chunk_text) // 2
        half = window_size // 2
        start = max(0, center - half)
        end = min(len(full_text), center + half)
        if start == 0:
            end = min(len(full_text), window_size)
        elif end == len(full_text):
            start = max(0, len(full_text) - window_size)
        snippet = full_text[start:end]
        if start > 0:
            snippet = "…" + snippet
        if end < len(full_text):
            snippet = snippet + "…"
        return snippet
    elif len(full_text) > window_size:
        return full_text[:window_size] + "…（场景略）"
    else:
        return full_text


def get_samples(base_dir):
    pattern = os.path.join(base_dir, "debug", "chunks", "*_parent_chunks.jsonl")
    files = glob.glob(pattern)
    chunks = []
    for f in files:
        with open(f, "r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    try:
                        data = json.loads(line)
                        if "full_ai_text" in data and len(data["full_ai_text"]) > 100:
                            chunks.append(data)
                    except:
                        pass

    if len(chunks) > 80:
        chunks = random.sample(chunks, 80)
    return chunks


def extract_subchunk(full_text):
    if len(full_text) < 50:
        return full_text
    # Pick a random 50-character slice to act as the "hit" sub-chunk
    start = random.randint(0, len(full_text) - 50)
    return full_text[start:start + 50]


async def llm_judge(client, text_a, text_b):
    if not API_KEY:
        # Dummy judge if no token locally (avoids crash, but user expects API call)
        # We will retrieve token later if needed.
        pass

    prompt = f"""你是一个高级评估裁判。我们正在进行对话上下文截取。
我们需要判断哪种截取方式更适合作为角色扮演(RP)的长期记忆上下文（信息保留完整、冗余少、截断自然）。

样本 A:
{text_a}

样本 B:
{text_b}

请仔细比较两者，最终在一行内仅输出 "A", "B", 或 "TIE"。"""

    try:
        response = await client.post(
            API_URL, 
            json={
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100,
                "temperature": 0.1
            },
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=15.0
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].upper()
        if "A" in content and "B" not in content[-3:]:
            return "A"
        elif "B" in content and "A" not in content[-3:]:
            return "B"
        else:
            return "TIE"
    except Exception as e:
        print(f"API Error: {e}")
        return "TIE"


async def main():
    base_dir = r"e:\Aegis_Isle\AegisIsle_cc_ver\Aegis-Isle"

    # Load .env
    env_path = os.path.join(base_dir, ".env")
    global API_KEY
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("OPENAI_API_KEY="):
                    API_KEY = line.strip().split("=", 1)[1].strip("'\"")

    if not API_KEY:
        print("WARNING: OPENAI_API_KEY not found. Calling API might fail.")

    chunks = get_samples(base_dir)
    print(f"Sampled {len(chunks)} chunks.")

    # Initialize scores. Wins track.
    scores = {ws: 0 for ws in WINDOW_SIZES}
    total_matches_per_ws = {ws: 0 for ws in WINDOW_SIZES}

    async with httpx.AsyncClient() as client:
        for idx, chunk in enumerate(chunks):
            print(f"Processing chunk {idx+1}/{len(chunks)}")
            full_text = chunk["full_ai_text"]
            meta_str = " | ".join(f"{k}: {v}" for k, v in chunk.get("scene_meta", {}).items() if v)
            user_msg = chunk.get("user_msg", "")

            sub_chunk = extract_subchunk(full_text)

            # Generate texts
            texts = {}
            for ws in WINDOW_SIZES:
                snippet = centered_extract(full_text, sub_chunk, ws)
                parts = [f"[场景元数据: {meta_str}]"]
                parts.append(f"[User曾说]: {user_msg}")
                parts.append(f"[相关上下文]: {snippet}")
                texts[ws] = "\n".join(parts)

            # Pairwise compare
            for w1, w2 in combinations(WINDOW_SIZES, 2):
                await asyncio.sleep(2)  # rate limit requested by user
                winner = await llm_judge(client, texts[w1], texts[w2])

                total_matches_per_ws[w1] += 1
                total_matches_per_ws[w2] += 1

                if winner == "A":
                    scores[w1] += 1
                elif winner == "B":
                    scores[w2] += 1
                else:
                    scores[w1] += 0.5
                    scores[w2] += 0.5

    # Summarize
    report = []
    report.append("# WINDOW_SIZE 寻参结果报告")
    report.append(f"**测试样本数量**: {len(chunks)}")
    report.append(f"**测试窗口梯度**: {WINDOW_SIZES}")
    report.append("---")
    report.append("## 胜率/得分统计")

    best_ws = None
    best_score = -1
    for ws in WINDOW_SIZES:
        match_count = max(1, total_matches_per_ws[ws])
        avg = (scores[ws] / match_count) * 100
        report.append(f"- **WINDOW_SIZE {ws}**: {avg:.2f} 分 (胜局: {scores[ws]}/{match_count})")
        if avg > best_score:
            best_score = avg
            best_ws = ws

    report.append("---")
    report.append(f"## 🏆 推荐默认值: **{best_ws}**")

    with open(os.path.join(base_dir, "cowokers_ai", "WINDOW_SIZE_RESULT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"Done. Recommended WINDOW_SIZE: {best_ws}")

    # Append to NIGHTLY_DONE.md
    done_path = os.path.join(base_dir, "cowokers_ai", "NIGHTLY_DONE.md")
    with open(done_path, "a", encoding="utf-8") as f:
        f.write(f"\n✅ WINDOW_SIZE 寻参完成 - 推荐值: [{best_ws}] - 结果文件: cowokers_ai/WINDOW_SIZE_RESULT.md\n")

if __name__ == "__main__":
    asyncio.run(main())
