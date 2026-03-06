/**
 * Aegis-Isle 长线记忆 SillyTavern 插件 v0.4.0
 */

// ← 这行日志用来确认脚本被加载进来了
console.log('🧠 [AegisMemory] 脚本文件已加载，开始初始化...');

(function () {
    'use strict';

    const MODULE_NAME = 'aegis_memory';
    const DEFAULT_SETTINGS = {
        enabled: true,
        aegis_base_url: 'http://127.0.0.1:8001',
        character_name: '',
        world_line: '',
        k: 3,
        realtime_ingest: true,
        debug: false,
    };

    // ============================================================
    // 设置管理 —— 用 SillyTavern.getContext() 存설置
    // ============================================================

    function getSettings() {
        try {
            const ctx = SillyTavern.getContext();
            const es = ctx.extensionSettings;
            if (!es[MODULE_NAME]) {
                es[MODULE_NAME] = Object.assign({}, DEFAULT_SETTINGS);
            }
            // 补齐新字段
            for (const k of Object.keys(DEFAULT_SETTINGS)) {
                if (!(k in es[MODULE_NAME])) es[MODULE_NAME][k] = DEFAULT_SETTINGS[k];
            }
            return es[MODULE_NAME];
        } catch (e) {
            return Object.assign({}, DEFAULT_SETTINGS);
        }
    }

    function saveSettings() {
        try { SillyTavern.getContext().saveSettingsDebounced(); } catch (e) { /* 静默 */ }
    }

    function getCurrentChar() {
        const s = getSettings();
        if (s.character_name) return s.character_name;
        try { return SillyTavern.getContext().name2 || 'default_char'; }
        catch (e) { return 'default_char'; }
    }

    // ============================================================
    // Aegis 后端 API
    // ============================================================

    async function queryMemory(query) {
        const s = getSettings();
        try {
            const res = await fetch(`${s.aegis_base_url}/v1/memory/search`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: query.slice(0, 500),
                    character_name: getCurrentChar(),
                    world_line: s.world_line || null,
                    k: s.k,
                }),
                signal: AbortSignal.timeout(5000),
            });
            if (!res.ok) return null;
            const data = await res.json();
            console.log('[AegisMemory] 后端网关查询完成, 查到 FAISS 记忆片段:', data.count);
            return {
                context_string: data.context_string || null,
                debug_info: data.debug_info || null
            };
        } catch (err) {
            console.log('[AegisMemory] 查询记忆失败（不影响聊天）:', err.message);
            return null;
        }
    }

    async function ingestMemory(userMsg, aiMsg) {
        const s = getSettings();
        if (!s.realtime_ingest || !userMsg || !aiMsg) return;
        const charName = getCurrentChar();
        let userName = 'User', charDisplay = charName;
        try { const c = SillyTavern.getContext(); userName = c.name1 || 'User'; charDisplay = c.name2 || charName; }
        catch (e) { /* 静默 */ }
        try {
            await fetch(`${s.aegis_base_url}/v1/memory/ingest`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    character_name: charName,
                    world_line: s.world_line || null,
                    messages: [
                        { role: 'user', name: userName, content: userMsg },
                        { role: 'char', name: charDisplay, content: aiMsg },
                    ],
                    chat_file: 'realtime',
                }),
                signal: AbortSignal.timeout(8000),
            });
        } catch (err) { /* 静默 */ }
    }

    // ============================================================
    // 预取记忆机制（核心修复）
    // ============================================================
    //
    // 问题：CHAT_COMPLETION_PROMPT_READY 是同步事件，SillyTavern 不等
    //       async handler 的 Promise，所以 await fetch() 永远"迟到"。
    //
    // 解法：用户点击发送按钮时就立刻开始 fetch（pre-fetch），
    //       把 Promise 存到 prefetchPromise。
    //       CHAT_COMPLETION_PROMPT_READY 触发时再 await 这个已经
    //       在途中的 Promise，通常 67ms 已经完成，注入就变同步了。
    // ============================================================

    let lastUserMsg    = '';
    let prefetchQuery  = '';   // 预取时用的 query
    let prefetchPromise = null; // 预取中的 Promise<string|null>
    let injectedThisTurn = false; // 防重入

    function startPrefetch(query) {
        if (!query || query === prefetchQuery) return;
        prefetchQuery   = query;
        prefetchPromise = queryMemory(query);
        injectedThisTurn = false;
        console.log('[AegisMemory] 预取记忆已启动 query:', query.slice(0, 30));
    }

    // 监听发送按钮点击 & Enter 键，立刻开始 pre-fetch
    function setupSendListener() {
        // 发送按钮
        const sendBtn = document.getElementById('send_but');
        if (sendBtn) {
            sendBtn.addEventListener('click', () => {
                const ta = document.getElementById('send_textarea');
                const q  = ta?.value?.trim() || '';
                if (q) startPrefetch(q);
            });
        }

        // Enter 键（主输入框）
        const textarea = document.getElementById('send_textarea');
        if (textarea) {
            textarea.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    const q = textarea.value?.trim() || '';
                    if (q) startPrefetch(q);
                }
            });
        }
    }

    function setupListeners() {
        try {
            const ctx = SillyTavern.getContext();
            const { eventSource, event_types } = ctx;
            if (!eventSource || !event_types) {
                console.warn('[AegisMemory] eventSource 不可用');
                return;
            }

            // ★ Hook 1：发消息前注入记忆（await 已在途的 prefetch Promise）
            eventSource.on(event_types.CHAT_COMPLETION_PROMPT_READY, async (eventData) => {
                const s = getSettings();
                if (!s.enabled || !eventData?.chat) return;
                if (injectedThisTurn) {
                    console.log('[AegisMemory] 跳过重复注入');
                    return;
                }

                const chat = eventData.chat;

                // 如果 prefetchPromise 还没启动，临时用当前 query 发起
                if (!prefetchPromise) {
                    const lastUser = [...chat].reverse().find(m => m.role === 'user');
                    const q = typeof lastUser?.content === 'string'
                        ? lastUser.content
                        : (lastUser?.content || []).map(c => c.text || '').join(' ');
                    lastUserMsg = q;
                    if (q) startPrefetch(q);
                }

                if (!prefetchPromise) return;

                injectedThisTurn = true;
                const memResult = await prefetchPromise;
                prefetchPromise = null; // 消耗掉

                if (!memResult || !memResult.context_string) return;

                const memCtx = memResult.context_string;
                const debugInfo = memResult.debug_info;

                if (debugInfo) {
                    console.log('%c🔍 [Aegis 网关路由详情]', 'color: #00bcd4; font-weight: bold;');
                    console.log(`  🌐 FAISS 向量检索: ${debugInfo.routed_faiss ? '命中 (长度 '+debugInfo.faiss_len+')' : '未命中'}`);
                    console.log(`  🌐 Graph 实体检索: ${debugInfo.routed_graph ? '命中 (长度 '+debugInfo.graph_len+')' : '未命中'}`);
                    console.log(`  🌐 Episode 剧情检索: ${debugInfo.routed_episode ? '命中 (长度 '+debugInfo.episode_len+')' : '未命中'}`);
                }

                const lastUser = [...chat].reverse().find(m => m.role === 'user');
                if (lastUser?.content) {
                    lastUserMsg = typeof lastUser.content === 'string'
                        ? lastUser.content
                        : (lastUser.content || []).map(c => c.text || '').join(' ');
                }

                console.log('[AegisMemory] ✅ 注入混合上下文记忆到 Prompt（预取模式）！');
                
                // 检查有没有占位符拦截（任务 C: /recap 等）这可以在前端执行提示
                if (lastUserMsg && lastUserMsg.startsWith('/')) {
                   console.log('[AegisMemory] 检测到指令输入: ', lastUserMsg);
                }

                // 【核心修复 v2】将记忆作为独立 system 消息插入到最后一条 user 消息之前（depth=1）
                // 不再拼在 user 消息的 content 里，避免截断紧邻的 [system] 标签
                const lastUserIdx = chat.map(m => m.role).lastIndexOf('user');
                if (lastUserIdx >= 0) {
                    chat.splice(lastUserIdx, 0, {
                        role: 'system',
                        content: memCtx,
                        identifier: 'aegis_memory_injection'
                    });
                }
                
                // DEBUG_SAVE 功能（任务 H）：把最终组装完的所有 payload 发往后端持久化
                if (s.debug) {
                    try {
                        const promptText = chat.map(c => `[${c.role}]: ${typeof c.content === 'string' ? c.content : JSON.stringify(c.content)}`).join('\\n\\n');
                        const charAndWorld = s.world_line ? `${getCurrentChar()}_${s.world_line}` : getCurrentChar();
                        await fetch(`${s.aegis_base_url}/v1/memory/debug_save`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                universe_id: charAndWorld,
                                prompt_text: promptText,
                            })
                        });
                        console.log('[AegisMemory] 最终拼接块已发送给后端保存至 debug/prompts');
                    } catch (e) {
                         console.log('[AegisMemory] 最终拼接收集发送失败:', e);
                    }
                }

                // 重置锁，供下一轮使用
                setTimeout(() => { injectedThisTurn = false; }, 1000);
            });

            // ★ Hook 2：AI 回复后，实时存入新记忆
            eventSource.on(event_types.MESSAGE_RECEIVED, (data) => {
                const s = getSettings();
                if (!s.enabled || !s.realtime_ingest) return;
                const aiMsg = data?.mes || '';
                if (lastUserMsg && aiMsg) {
                    setTimeout(() => {
                        ingestMemory(lastUserMsg, aiMsg);
                        lastUserMsg = '';
                    }, 0);
                }
            });

            console.log('[AegisMemory] 消息监听器已注册（CHAT_COMPLETION_PROMPT_READY + MESSAGE_RECEIVED）');
        } catch (e) {
            console.error('[AegisMemory] 消息监听器注册失败:', e.message);
        }
    }


    // ============================================================
    // 设置面板 UI
    // ============================================================

    function renderUI() {
        console.log('[AegisMemory] 渲染设置面板...');
        const s = getSettings();

        const html = `
<div id="aegis-memory-panel">
  <div class="inline-drawer">
    <div class="inline-drawer-toggle inline-drawer-header">
      <b>🧠 Aegis-Isle 长线记忆</b>
      <div class="inline-drawer-icon fa-solid fa-circle-chevron-down down"></div>
    </div>
    <div class="inline-drawer-content">
      <label style="display:block;color:#aaa;font-size:.8em;margin-top:6px">Aegis 后端地址</label>
      <input id="aegis-url" type="text" value="${s.aegis_base_url}" placeholder="http://127.0.0.1:8001" style="width:100%;box-sizing:border-box;margin-bottom:5px;" />
      <label style="display:block;color:#aaa;font-size:.8em">角色 ID（留空用角色卡名）</label>
      <input id="aegis-char" type="text" value="${s.character_name}" placeholder="ZouZheng" style="width:100%;box-sizing:border-box;margin-bottom:5px;" />
      <label style="display:block;color:#aaa;font-size:.8em">世界线分组（可选）</label>
      <input id="aegis-world" type="text" value="${s.world_line}" placeholder="AIDom" style="width:100%;box-sizing:border-box;margin-bottom:8px;" />
      <label><input id="aegis-enabled" type="checkbox" ${s.enabled ? 'checked' : ''} /> 启用长线记忆注入</label><br/>
      <label><input id="aegis-realtime" type="checkbox" ${s.realtime_ingest ? 'checked' : ''} /> 实时存入新对话到记忆库</label><br/>
      <label><input id="aegis-debug" type="checkbox" ${s.debug ? 'checked' : ''} /> Debug 日志</label>
      <div id="aegis-status" style="font-size:.8em;margin-top:8px;color:${s.enabled ? '#4caf50' : '#888'}">
        ${s.enabled ? '🟢 长线记忆已激活' : '⚫ 已停用'}
      </div>
    </div>
  </div>
</div>`;

        // 尝试多个容器
        const containers = [
            document.getElementById('extensions_settings'),
            document.getElementById('extensions_settings2'),
            document.querySelector('.extensions_settings'),
        ];
        const target = containers.find(el => el !== null);

        if (target) {
            target.insertAdjacentHTML('beforeend', html);
            console.log('[AegisMemory] 面板已挂载到:', target.id || target.className);
        } else {
            console.warn('[AegisMemory] 未找到扩展设置容器！等待...');
            // 再试一次
            setTimeout(() => {
                const t2 = document.getElementById('extensions_settings') ||
                           document.getElementById('extensions_settings2');
                if (t2) { t2.insertAdjacentHTML('beforeend', html); bindEvents(); }
            }, 2000);
            return;
        }

        bindEvents();
    }

    function bindEvents() {
        const bind = (id, key, isCheckbox = false) => {
            const el = document.getElementById(id);
            if (!el) return;
            el.addEventListener('change', function () {
                const s = getSettings();
                s[key] = isCheckbox ? this.checked : this.value.trim();
                if (key === 'enabled') {
                    const statusEl = document.getElementById('aegis-status');
                    if (statusEl) {
                        statusEl.style.color = s.enabled ? '#4caf50' : '#888';
                        statusEl.textContent = s.enabled ? '🟢 长线记忆已激活' : '⚫ 已停用';
                    }
                }
                saveSettings();
            });
        };
        bind('aegis-url', 'aegis_base_url');
        bind('aegis-char', 'character_name');
        bind('aegis-world', 'world_line');
        bind('aegis-enabled', 'enabled', true);
        bind('aegis-realtime', 'realtime_ingest', true);
        bind('aegis-debug', 'debug', true);
    }

    // ============================================================
    // 初始化入口 —— 使用原生 DOMContentLoaded，不依赖 jQuery
    // ============================================================

    function init() {
        console.log('[AegisMemory] DOM ready，开始初始化...');
        renderUI();
        setupListeners();
        setupSendListener();
        console.log('[AegisMemory] ✅ 插件初始化成功！后端:', getSettings().aegis_base_url);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        // DOM 已经 ready 了（懒加载场景）
        init();
    }

})();
