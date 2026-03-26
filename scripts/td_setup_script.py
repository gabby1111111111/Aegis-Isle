# flake8: noqa
# TouchDesigner一键自动构建脚本：Bubby (阶段一)
# 作者：Antigravity
# 说明：将此代码复制到 TouchDesigner 的 Textport 中运行，或者在一个新创建的 Text DAT 中右键点击 "Run Script" 即可自动生成完整的 Bubby 视觉节点树和 WebSocket 接收逻辑。

import json
try:
    from td import *
except ImportError:
    pass # for flake8 linting outside TD

def build_bubby():
    # 改为放在 project1，这是 TD 默认打开能看到的地方！之前放在根目录导致您在默认界面看不到它。
    root = op('/project1/Bubby')
    if not root:
        root = op('/project1').create(baseCOMP, 'Bubby')
    
    # ----------------------------------------------------
    # 1. 通信层：WebSocket DAT
    # ----------------------------------------------------
    # 尝试查找已有的，如果没有则创建
    ws = root.op('st_ws')
    if not ws:
        ws = root.create(websocketDAT, 'st_ws')

    # ----------------------------------------------------
    # 2. 几何体层：Sphere + Noise
    # ----------------------------------------------------
    sphere = root.op('bubby_body')
    if not sphere:
        sphere = root.create(sphereSOP, 'bubby_body')
    
    noise = root.op('bubby_noise')
    if not noise:
        noise = root.create(noiseSOP, 'bubby_noise')
    
    # 将 Sphere 连接到 Noise
    try:
        noise.inputConnectors[0].connect(sphere)
    except:
        pass

    # ----------------------------------------------------
    # 3. 材质层：PBR 透明虹彩材质
    # ----------------------------------------------------
    mat = root.op('bubby_mat')
    if not mat:
        mat = root.create(pbrMAT, 'bubby_mat')
        
    # 将材质应用给几何体
    try:
        noise.par.material = mat.path
    except:
        pass

    # ----------------------------------------------------
    # 4. 文本层：思维闪现与残影、下落效果
    # ----------------------------------------------------
    text = root.op('thought_text')
    if not text:
        text = root.create(textTOP, 'thought_text')

    # --- 新增的下落动画层 ---
    drop = root.op('text_drop')
    if not drop:
        drop = root.create(transformTOP, 'text_drop')
    try:
        drop.inputConnectors[0].connect(text)
    except:
        pass

    feedback = root.op('thought_trail')
    if not feedback:
        feedback = root.create(feedbackTOP, 'thought_trail')
    try:
        feedback.inputConnectors[0].connect(drop)
        feedback.par.targetop = feedback.name
    except:
        pass

    level = root.op('trail_fade')
    if not level:
        level = root.create(levelTOP, 'trail_fade')
    try:
        trail_move = root.op('trail_move')
        if not trail_move:
            trail_move = root.create(transformTOP, 'trail_move')
        trail_move.inputConnectors[0].connect(level)
        
        level.inputConnectors[0].connect(feedback)
        feedback.par.targetop = trail_move.name
    except:
        pass

    comp = root.op('trail_comp')
    if not comp:
         comp = root.create(overTOP, 'trail_comp')
    try:
        # 背景为残影，前景为下落的当前帧文字
        comp.inputConnectors[0].connect(drop)
        comp.inputConnectors[1].connect(trail_move)
    except:
        pass

    # ----------------------------------------------------
    # 5. 渲染层：Geometry, Camera, Light, Render
    # ----------------------------------------------------
    geo = root.op('bubby_geo')
    if not geo:
        geo = root.create(geometryCOMP, 'bubby_geo')
    try:
        geo.inputConnectors[0].connect(noise)  # 将噪波几何体连入 Geo
    except:
        pass
        
    cam = root.op('bubby_cam')
    if not cam:
        cam = root.create(cameraCOMP, 'bubby_cam')

    light = root.op('bubby_light')
    if not light:
        light = root.create(lightCOMP, 'bubby_light')

    render = root.op('bubby_render')
    if not render:
        render = root.create(renderTOP, 'bubby_render')

    # 最终合成：将 3D 的泡泡背景与前端飘落的文字痕迹叠在一起
    final_output = root.op('final_output')
    if not final_output:
        final_output = root.create(overTOP, 'final_output')
    try:
        # Background: bubby_render (泡泡)
        # Foreground: trail_comp (文字和残影)
        final_output.inputConnectors[0].connect(comp)
        final_output.inputConnectors[1].connect(render)
    except:
        pass

    # ----------------------------------------------------
    # 6. WebSocket 接收与平滑过渡逻辑 (DAT)
    # ----------------------------------------------------
    ws_cb = root.op('ws_callbacks')
    if not ws_cb:
         ws_cb = root.create(textDAT, 'ws_callbacks')
    
    # 编写回调脚本
    script_content = """# WebSocket 接收逻辑
import json

def onReceiveText(dat, rowIndex, message):
    try:
        data = json.loads(message)
        
        # 1. 提取目标数据
        mood = data.get('mood_intensity', 0.3)
        color = data.get('base_color', [0.5, 0.7, 0.9])
        txt = data.get('thought_text', '')
        
        # 2. 获取目标节点
        noise = op('bubby_noise')
        mat = op('bubby_mat')
        text = op('thought_text')
        drop = op('text_drop')
        
        # 3. 映射到参数
        # 噪音振幅和频率随情绪增加
        noise.par.amplitude = 0.05 + (mood * 0.15)
        noise.par.period = 2.0 - (mood * 1.0)
        
        # 基础颜色修改
        mat.par.basecolorr = color[0]
        mat.par.basecolorg = color[1]
        mat.par.basecolorb = color[2]
        
        # 显示思考文本并触发下落
        if txt:
            text.par.text = txt
            text.par.colora = 1.0
            
            # 使用 store 记录当前触发的时间戳
            drop.store('start_time', absTime.seconds)
            
            # Y 坐标下落表达式：从 0.2 (屏幕中上) 开始，以每秒 0.15 的速度往下降落
            drop.par.ty.expr = "0.2 - (absTime.seconds - me.fetch('start_time', absTime.seconds)) * 0.15"
            
            # 自动淡出文本 (1.5秒后完全透明)，残影还会保留一阵子
            run('op("thought_text").par.colora = 0.0', delayFrames=90)
            
    except Exception as e:
        print("TD 解析 ST 数据出错:", e)
"""
    ws_cb.text = script_content
    # 绑定回调
    ws.par.callbacks = ws_cb.name

    print("============ Bubby 节点生成完毕 ============")
    print("1. 确保 st_ws(websocketDAT) 已连上服务器")
    print("2. 您可能需要手动添加 Camera 和 Render TOP 查看最终画面")

# 运行构建
build_bubby()
