import streamlit as st
import json
import os
from pathlib import Path

st.set_page_config(page_title="CharLife 审核面板", page_icon="📝", layout="centered")

st.title("📝 CharLifeAgent 白天感悟审核")
st.markdown("审查角色自治思考的碎片。**批准**的条目将在夜间 DailyDigest 管线中被编译写入 FAISS；**驳回**的条目将被直接遗忘。")

EVENTS_DIR = Path("data/diary/events")
EVENTS_DIR.mkdir(parents=True, exist_ok=True)
PENDING_FILE = EVENTS_DIR / "pending_char_activity.jsonl"
APPROVED_FILE = EVENTS_DIR / "character_activity.jsonl"

def load_events(file_path):
    if not file_path.exists():
        return []
    events = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except:
                    pass
    return events

def save_events(file_path, events):
    with open(file_path, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")

def append_event(file_path, event):
    with file_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

def approve_event(index):
    ev = st.session_state.pending_events[index]
    append_event(APPROVED_FILE, ev)
    st.session_state.pending_events.pop(index)
    save_events(PENDING_FILE, st.session_state.pending_events)

def reject_event(index):
    st.session_state.pending_events.pop(index)
    save_events(PENDING_FILE, st.session_state.pending_events)

# 初始化状态
if "pending_events" not in st.session_state:
    st.session_state.pending_events = load_events(PENDING_FILE)

# 如果 pending 为空，尝试从旧系统兼容（将 character_activity 暂存的内容当作待审，但不强制，仅当用户刚执行过 CharLifeAgent 时可能发生这种情况）
# 但由于避免误操作影响 DailyDigest，我们只专注处理 pending_FILE。

events = st.session_state.pending_events

if not events:
    st.success("🎉 当前没有需要审核的自治日记！")
    
    if st.button("刷新状态", type="primary"):
        st.session_state.pending_events = load_events(PENDING_FILE)
        st.rerun()
else:
    st.info(f"待审核队列中共有 **{len(events)}** 条记录。")
    
    for i, ev in enumerate(events):
        details = ev.get('details', {})
        character = ev.get('character', 'Unknown')
        topic = details.get('source_topic', '无主题')
        reaction = details.get('char_reaction', '（无记录）')
        emotion = details.get('emotion_tag', '平静')
        timestamp = ev.get('timestamp', '刚刚')
        
        with st.container(border=True):
            st.subheader(f"💡 {character} 的内心独白")
            st.caption(f"**生成时间**: {timestamp}")
            st.markdown(f"**触发源**: `{topic}`")
            st.info(f"\"{reaction}\"")
            st.markdown(f"**情绪推断**: `{emotion}`")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ 批准写入 Diary FAISS", key=f"approve_{i}", use_container_width=True):
                    approve_event(i)
                    st.rerun()
            with col2:
                if st.button("❌ 驳回", key=f"reject_{i}", use_container_width=True):
                    reject_event(i)
                    st.rerun()
                    
        # 为了防误操作，每次只展示第一条（类似于 Tinder 审查卡片）
        break
