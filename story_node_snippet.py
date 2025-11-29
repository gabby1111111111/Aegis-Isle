# 添加剧情节点渲染函数到 interview_app.py

def render_story_node():
    """Render a story node (cinematic moment)."""
    story_trigger = st.session_state.pending_story_node
    
    if not story_trigger:
        return
    
    # Get trigger description
    trigger_info = st.session_state.story_manager.triggers.get(story_trigger)
    
    st.markdown(f"""
    <div class="cinematic-box" style="border: 2px solid #ffd700; background: linear-gradient(180deg, #1a0a0a 0%, #000000 100%);">
        <h2 style="text-align: center; color: #ffd700;">🌟 {trigger_info.description if trigger_info else '剧情节点'} 🌟</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate story content
    success_rate = st.session_state.story_manager.get_success_rate()
    
    # Determine node type based on trigger
    if "box_1" in story_trigger:
        node_type = "node_a"
        title = "🧬 初次觉醒"
    elif "box_3" in story_trigger:
        node_type = "node_b"  
        title = "⚔️ 晋升试炼"
    else:
        node_type = "mastery"
        title = "👑 荣誉时刻"
    
    # Generate story with async
    story_placeholder = st.empty()
    
    async def generate_story():
        story_data = await st.session_state.generator.generate_story_node(
            st.session_state.current_persona,
            node_type,
            success_rate,
            language=st.session_state.language
        )
        
        story_content = story_data.get("story_content", "剧情生成中...")
        
        story_placeholder.markdown(f"""
        <div class="cinematic-box">
            <h3 style="color: #ff6b9d;">{title}</h3>
            <p style="font-size: 18px; line-height: 1.8;">{story_content}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Run async generation
    import asyncio
    asyncio.run(generate_story())
    
    # Button to continue
    if st.button("继续修行", key="continue_from_story"):
        st.session_state.pending_story_node = None
        asyncio.run(generate_new_question())
        st.rerun()
