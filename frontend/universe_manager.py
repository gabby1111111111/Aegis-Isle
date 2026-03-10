"""
Aegis-Isle 世界线管理器 (Universe Manager)
一个 Streamlit 面板，用于：
  - 查看角色的所有世界线（宇宙）
  - 浏览每个宇宙的 episode 章节摘要与原文
  - 给世界线重命名 / 自动取名
  - 语义搜索 + 对返回 chunk 打分 (Re-ranking 反馈闭环)
"""

import streamlit as st
import httpx
import json

# ─────────────────────────────────────────
# 页面基本配置
# ─────────────────────────────────────────
st.set_page_config(
    page_title="🌌 Aegis 世界线管理器",
    page_icon="🌌",
    layout="wide",
)

AEGIS_URL = "http://127.0.0.1:8001/v1"


def api(method: str, path: str, **kwargs):
    """封装 API 调用，带超时与错误提示"""
    try:
        r = httpx.request(method, f"{AEGIS_URL}{path}", timeout=60, **kwargs)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"❌ API 请求失败: {e}")
        return None


# ─────────────────────────────────────────
# 侧边栏：角色选择与世界线列表
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎭 角色 & 宇宙")
    character = st.text_input("角色名", value="邹峥", help="输入角色名来加载该角色的所有世界线")

    if st.button("🔄 刷新", use_container_width=True):
        st.cache_data.clear()

    if character:
        data = api("GET", f"/universe/list?character={character}")
        universes = data.get("universes", []) if data else []

        if not universes:
            st.warning("未找到该角色的任何世界线，请先导入聊天记录。")
            selected_universe = None
        else:
            # 将 alias 展示在下拉菜单里
            alias_map = {u["alias"]: u for u in universes}
            selected_alias = st.selectbox(
                f"📂 世界线 ({len(universes)} 个)",
                list(alias_map.keys()),
            )
            selected_universe = alias_map[selected_alias]
    else:
        universes = []
        selected_universe = None


# ─────────────────────────────────────────
# 主内容区：Tabs
# ─────────────────────────────────────────
st.markdown("# 🌌 Aegis-Isle 世界线管理器")

if selected_universe is None:
    st.info("← 请在左侧输入角色名并选择一个世界线。")
    st.stop()

tab_overview, tab_chapters, tab_search, tab_clean = st.tabs([
    "📋 概览 & 设置",
    "📖 章节浏览",
    "🔍 语义搜索 & 打分",
    "🧹 数据清洗诊断"
])


# ─────────────────────────────────────────
# Tab 1：概览 & 设置
# ─────────────────────────────────────────
with tab_overview:
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader(f"🌐 {selected_universe['alias']}")
        st.caption(f"内部 ID: `{selected_universe['universe_id']}`")

        # 显示已知的真实信息（从 episodes 聚合）
        st.markdown("**📌 世界线概要**")
        m0, m1, m2 = st.columns(3)
        m0.metric("角色", selected_universe.get("character_name", "?"))
        m1.metric("章节数", selected_universe["episode_count"])
        m2.metric("记忆切片", selected_universe["chunk_count"])

        time_start = selected_universe.get("time_start", "")
        serial_start = selected_universe.get("serial_start", "")
        if time_start:
            st.caption(f"⏱️ 故事起点: `{serial_start}`  |  `{time_start}`")

    with col_right:
        st.markdown("**✏️ 重命名世界线**")
        new_alias = st.text_input("新名称", value=selected_universe["alias"])
        if st.button("保存重命名", use_container_width=True):
            res = api("POST", "/universe/rename", json={
                "universe_id": selected_universe["universe_id"],
                "alias": new_alias
            })
            if res:
                st.success(f"✅ 已重命名为「{new_alias}」，请刷新侧边栏。")

        st.markdown("---")
        st.markdown("**🤖 AI 自动取名**")
        st.caption("根据所有章节摘要让 LLM 生成一个文学标题")
        if st.button("✨ 自动生成名称", use_container_width=True):
            with st.spinner("AI 正在思考..."):
                res = api("POST", f"/universe/auto_name?universe_id={selected_universe['universe_id']}")
            if res and res.get("status") == "ok":
                st.success(f"✅ 生成名称：「{res['alias']}」（方式: {res.get('method', 'llm')}）")
            else:
                st.error("自动取名失败，请检查后端日志。")


# ─────────────────────────────────────────
# Tab 2：章节浏览（Episode + Sub-chunks）
# ─────────────────────────────────────────
with tab_chapters:
    ep_data = api("GET", f"/universe/episodes?universe_id={selected_universe['universe_id']}")
    episodes = ep_data.get("episodes", []) if ep_data else []

    if not episodes:
        st.warning("该世界线暂无 episode 数据。")
    else:
        st.markdown(f"**共 {len(episodes)} 个章节**")

        for ep in episodes:
            label = f"📖 {ep.get('serial', ep.get('episode_id', '?'))}  ·  `{ep.get('time_range', '')}`"
            with st.expander(label, expanded=False):
                # 摘要部分
                st.markdown("**📝 章节摘要**")
                st.markdown(ep.get("plot", "（无摘要）"))

                # 场景
                if ep.get("scene"):
                    st.caption(f"🎬 场景：{ep['scene']}")

                # 伏笔 Seeds — 改用 markdown 代码块避免嵌套 expander 报错
                seeds = ep.get("seeds", [])
                if seeds:
                    seeds_md = "\n".join(f"- {s}" for s in seeds)
                    st.markdown(
                        f"<details><summary>🌱 伏笔线索 ({len(seeds)} 条)</summary>\n\n{seeds_md}\n\n</details>",
                        unsafe_allow_html=True
                    )

                st.markdown("---")

                # 原文 sub_chunks 按需加载 —— 通过 scene_id 精确过滤到这一章
                ep_id = ep["episode_id"]
                ep_scene_id = ep.get("scene_id") or ep_id  # fallback to episode_id

                if st.button(f"📄 加载本章原文切片", key=f"load_{ep_id}"):
                    # 先拉全部 parent_chunks，找到序号匹配的父块
                    pc_data = api("GET", f"/universe/parent_chunks?universe_id={selected_universe['universe_id']}")
                    parent_chunks = pc_data.get("parent_chunks", []) if pc_data else []

                    # episode_id 末尾 3 位数字就是章节序号，例如 ep_..._001 -> 001
                    # parent.scene_id 末尾规律一致，例如 ..._scene_001 -> 001
                    ep_serial_num = ep_id.split('_')[-1].zfill(3)  # '001'

                    matched_parents = [
                        pc for pc in parent_chunks
                        if pc.get('scene_id', '').split('_')[-1].zfill(3) == ep_serial_num
                    ]

                    if matched_parents:
                        pc = matched_parents[0]
                        sub_ids = pc.get("sub_chunk_ids", [])
                        st.markdown(f"**📄 本章原文切片** (共 {len(sub_ids)} 片)")
                        st.caption(f"👤 用户发言: {pc.get('user_msg','')}")
                        meta = pc.get('scene_meta', {})
                        if meta:
                            col_m1, col_m2 = st.columns(2)
                            col_m1.caption(f"📅 {meta.get('date','')} {meta.get('time','')}")
                            col_m2.caption(f"🌤️ {meta.get('weather','')} | 📍{meta.get('location','')}")

                        # 加载该父块下的 sub_chunks
                        sc_data = api("GET",
                            f"/universe/chunks?universe_id={selected_universe['universe_id']}&parent_chunk_id={pc['parent_chunk_id']}"
                        )
                        sub_chunks = sc_data.get("chunks", []) if sc_data else []

                        if not sub_chunks:
                            st.info("未找到对应的子切片")
                        for i, sc in enumerate(sub_chunks, 1):
                            text = sc.get("text", "")
                            dh = sc.get("dh_index", "")
                            st.markdown(f"**切片 #{i}** (DH层级: `{dh}`)")
                            
                            # 使用原生的 HTML DIV 并在外层加上 markdown 处理可以实现漂亮的文本框，不受 st.text_area disabled 时灰暗字体的限制
                            text_html = text.replace('\n', '<br>')
                            st.markdown(
                                f"""<div style="padding: 15px; border-radius: 8px; background-color: #f7f9fc; border: 1px solid #e2e8f0; color: #1e293b; font-size: 1.05em; line-height: 1.6; margin-bottom: 20px;">
                                {text_html}
                                </div>""",
                                unsafe_allow_html=True
                            )
                    else:
                        st.warning(f"未找到对应父块（章节序号: {ep_serial_num}），共有 {len(parent_chunks)} 个父块")


# ─────────────────────────────────────────
# Tab 3：语义搜索 + 打分 (功能五 Re-ranking)
# ─────────────────────────────────────────
with tab_search:
    st.markdown("### 🔍 跨宇宙语义搜索")
    st.caption("搜索后，对每个返回的记忆切片打分，系统会在下次检索时优先推荐你评高分的内容。")

    # ── 意图种子词快捷按钮 ──
    seed_data = api("GET", "/universe/query_seed_phrases")
    seed_categories = seed_data.get("categories", {}) if seed_data else {}

    if seed_categories:
        st.markdown("**💭 快捷意图种子词**  <span style='color:#999; font-size:0.85em;'>点击将词语追加到你的 query 后面</span>",
                   unsafe_allow_html=True)

        if "query_prefix" not in st.session_state:
            st.session_state.query_prefix = ""

        for cat_name, phrases in seed_categories.items():
            with st.expander(f"**{cat_name}**", expanded=(cat_name == "回忆型")):
                cols_p = st.columns(min(len(phrases), 4))
                for pi, phrase in enumerate(phrases):
                    with cols_p[pi % 4]:
                        if st.button(phrase, key=f"seed_{cat_name}_{pi}",
                                    help=f"点击将 '{phrase}' 加入 query"):
                            st.session_state.query_prefix = phrase

    # 输入框：如果有 seed 被点击，自动填入
    prefix = st.session_state.get("query_prefix", "")
    query_default = prefix
    query = st.text_input("input query here",
                         value=query_default,
                         placeholder="例如：你还记得我们在书房里第一次上课吗...",
                         label_visibility="collapsed")
    st.caption("↑ 即将发送给向量库的 Query。想要加內容可以直接在上方编辑，点击快捷词会自动填入。")
    
    col_s1, col_s2, col_s3 = st.columns([2, 1, 1])
    with col_s1:
        # 支持限定要搜哪些宇宙
        options = list(alias_map.keys()) if 'alias_map' in locals() else []
        selected_search_universes = st.multiselect("🎯 限定搜索范围 (不选则搜索全宇宙)", options, default=[])
    with col_s2:
        k = st.number_input("条数", min_value=3, max_value=20, value=8)
    with col_s3:
        human_weight_pct = st.slider("人类打分权重 %", min_value=0, max_value=100, value=40, help="0%表示纯按向量相似度，100%表示纯按你给的评分排。默认40%")

    if st.button("🔍 开始搜索", use_container_width=True, type="primary") and query:
        with st.spinner("正在进行多路并发检索 + Re-ranking..."):
            
            target_uids = []
            if selected_search_universes:
                target_uids = [alias_map[a]["universe_id"] for a in selected_search_universes]
            uids_str = ",".join(target_uids)
            human_weight = human_weight_pct / 100.0

            req_url = f"/universe/search?query={query}&character={character}&k={k}&human_weight={human_weight}"
            if uids_str:
                req_url += f"&target_universes={uids_str}"

            res = api("GET", req_url)

        if res:
            results = res.get("results", [])
            searched = res.get("searched_universes", "?")
            msg = res.get("message", "")
            if not results:
                st.warning(f"未找到相关记忆片段。{msg}")
                if searched:
                    st.info(f"ℹ️ 已搜索 {searched} 个宇宙索引")
            else:
                st.success(f"✅ 找到 {len(results)} 个相关片段（跨 {searched} 个宇宙，已按 Re-ranking 评分排序）")

                for i, item in enumerate(results, 1):
                    with st.container():
                        cols = st.columns([5, 1, 1, 1])
                        with cols[0]:
                            st.markdown(f"**#{i}** | 🤖向量相似度: `{item['similarity']}` | 👤人类偏好: `{item['human_avg_score']}` | 🏆综合分: `{item['final_score']}`")
                            
                            text_html = item["text"].replace('\n', '<br>')
                            st.markdown(
                                f"""<div style="padding: 15px; border-radius: 8px; background-color: #f7f9fc; border: 1px solid #e2e8f0; color: #1e293b; font-size: 1.05em; line-height: 1.6; max-height: 250px; overflow-y: auto;">
                                {text_html}
                                </div>""",
                                unsafe_allow_html=True
                            )
                            chunk_world = item["metadata"].get("world_line", "未知宇宙")
                            st.caption(f"来源宇宙: `{chunk_world}` | chunk_id: `{item['chunk_id']}`")

                        with cols[1]:
                            st.markdown("<br><br>", unsafe_allow_html=True)
                            if st.button("👍", key=f"up_{i}", help="相关，记住它"):
                                api("POST", "/universe/feedback", json={
                                    "chunk_id": item["chunk_id"],
                                    "query": query,
                                    "score": 5
                                })
                                st.toast("已记录 ★★★★★")

                        with cols[2]:
                            st.markdown("<br><br>", unsafe_allow_html=True)
                            if st.button("😐", key=f"mid_{i}", help="一般"):
                                api("POST", "/universe/feedback", json={
                                    "chunk_id": item["chunk_id"],
                                    "query": query,
                                    "score": 3
                                })
                                st.toast("已记录 ★★★☆☆")

                        with cols[3]:
                            st.markdown("<br><br>", unsafe_allow_html=True)
                            if st.button("👎", key=f"down_{i}", help="不相关，降权"):
                                api("POST", "/universe/feedback", json={
                                    "chunk_id": item["chunk_id"],
                                    "query": query,
                                    "score": 1
                                })
                                st.toast("已记录 ★☆☆☆☆")

                        st.divider()


# ─────────────────────────────────────────
# Tab 4：数据清洗诊断
# ─────────────────────────────────────────
with tab_clean:
    st.markdown("### 🧹 ST 预设风格探针 & 清洗诊断")
    st.caption("将一段原始 ST 聊天记录粘贴进来，系统会自动识别您使用的预设风格并预览清洗后的效果。")

    raw_input = st.text_area(
        "📌 粘贴 AI 回复的原文（包括装饰符号、HTML、马克内容）",
        height=200,
        placeholder="这里粘贴任意一段您的 SillyTavern AI 回复原文...\n\n例如\uff1a\n*邹峥将销笔轻轻搞在笔托上。*\n「进来吧。」他开口了。\n``` html\n<div>装饰</div>\n```"
    )
    user_input = st.text_input("👤 用户发言（可选，助于辅助判断语境）", placeholder="我十二岁这年，和养父邹峥的初次懂面")

    if st.button("🔍 自动识别风格并清洗预览", type="primary", use_container_width=True) and raw_input:
        with st.spinner("风格探针中..."):
            result = api("POST", "/universe/diagnose_clean", json={
                "raw_text": raw_input,
                "user_msg": user_input
            })

        if result:
            style = result.get("style", {})
            st.markdown("---")

            # 风格识别结果
            col_r1, col_r2, col_r3 = st.columns(3)
            col_r1.metric("🎨 检测风格", style.get("style_name", "?"))
            col_r2.metric("🔥 清洗率", f"{result.get('clean_rate_pct', 0)}%")
            col_r3.metric("✅ 置信度", style.get("confidence_score", 0))

            st.caption(f"💡 {style.get('description', '')}")

            # 各风格识别分数
            scores = style.get("all_scores", {})
            if scores:
                with st.expander("📊 详细识别分数"):
                    for k_name, v_score in scores.items():
                        st.progress(min(v_score / 20, 1.0), text=f"{k_name}: {v_score}")

            # 建议
            recs = result.get("recommendations", [])
            if recs:
                st.markdown("**📢 清洗建议：**")
                for rec in recs:
                    st.info(rec)

            # 对比原文 vs 清洗后
            st.markdown("**🔍 清洗前后对比：**")
            col_b, col_a = st.columns(2)
            with col_b:
                st.caption(f"原文 ({result.get('original_length', 0)} 字)")
                orig_html = raw_input[:800].replace('\n', '<br>')
                st.markdown(
                    f"""<div style="padding:12px;background:#fff8f0;border:1px solid #fca;border-radius:8px;color:#333;font-size:0.95em;line-height:1.6;max-height:300px;overflow-y:auto;">{orig_html}</div>""",
                    unsafe_allow_html=True
                )
            with col_a:
                st.caption(f"清洗后 ({result.get('cleaned_length', 0)} 字)")
                clean_html = result.get("cleaned_preview", "").replace('\n', '<br>')
                st.markdown(
                    f"""<div style="padding:12px;background:#f0fff4;border:1px solid #6ee7b7;border-radius:8px;color:#1e293b;font-size:0.95em;line-height:1.6;max-height:300px;overflow-y:auto;">{clean_html}</div>""",
                    unsafe_allow_html=True
                )
