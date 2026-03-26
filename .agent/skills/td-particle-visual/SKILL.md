---
description: TouchDesigner 2026 粒子视觉生成 — 基于 GPU Instancing + MCP 远程控制的高级粒子球/流场效果全流程指南
---

# TouchDesigner 粒子视觉 Skill

> **优先级**：锦上添花 (Nice-to-have)，不要在此投入过多精力死磕。
> **核心原则**：先通过 MCP 查询参数，确认存在后再赋值。**绝不盲猜参数名。**

---

## 1. Antigravity 原生 MCP 工具（首选方式）

> [!IMPORTANT]
> **不要再手写 Python 脚本 + base64 编码 + POST！** Antigravity 自带 TD MCP 工具，直接调用即可。

### 标准工作流：先查后改，绝不盲猜

```
Step 1: mcp_touchdesigner_get_td_nodes        → 列出子节点，确认结构
Step 2: mcp_touchdesigner_get_td_node_parameters → 查看真实参数名
Step 3: mcp_touchdesigner_update_td_node_parameters → 精准修改
Step 4: mcp_touchdesigner_get_td_node_errors   → 自动检查报错
```

### 可用工具速查表

| 工具 | 用途 | 示例参数 |
|------|------|---------|
| `mcp_touchdesigner_get_td_nodes` | 列出子节点 | `parentPath="/project1/Bubby"` |
| `mcp_touchdesigner_get_td_node_parameters` | **查参数名**（最重要！） | `nodePath="/project1/Bubby/particle_geo"` |
| `mcp_touchdesigner_create_td_node` | 创建节点 | `parentPath="/project1/Bubby", nodeType="geometryCOMP", nodeName="particle_geo"` |
| `mcp_touchdesigner_update_td_node_parameters` | 更新参数 | `nodePath="...", properties={"instancing": 1}` |
| `mcp_touchdesigner_delete_td_node` | 删除节点 | `nodePath="/project1/Bubby/particle_geo/torus1"` |
| `mcp_touchdesigner_get_td_node_errors` | 检查报错 | `nodePath="/project1/Bubby"` |
| `mcp_touchdesigner_execute_python_script` | 执行复杂脚本（仅在需要循环/连线时用） | `script="..."` |
| `mcp_touchdesigner_get_td_class_details` | 查 TD 类的方法和属性 | `className="noiseSOP"` |
| `mcp_touchdesigner_get_td_module_help` | 获取模块帮助文档 | `moduleName="scatterSOP"` |
| `mcp_touchdesigner_get_td_info` | 获取 TD 版本信息 | 无参数 |

### 典型用法示例

**查参数（避免猜错）：**
```
mcp_touchdesigner_get_td_node_parameters(nodePath="/project1/Bubby/particle_geo")
→ 返回所有参数名和当前值，包括 instancing, instancesop, instancetx 等
```

**创建节点：**
```
mcp_touchdesigner_create_td_node(
    parentPath="/project1/Bubby/bubby_geo",
    nodeType="scatterSOP",
    nodeName="src_scatter"
)
```

**更新参数：**
```
mcp_touchdesigner_update_td_node_parameters(
    nodePath="/project1/Bubby/particle_geo",
    properties={"instancing": 1, "instancesop": "/project1/Bubby/bubby_geo/src_noise"}
)
```

**复杂操作（需要连线/循环时才用 execute_python_script）：**
```
mcp_touchdesigner_execute_python_script(
    script="op('/project1/Bubby/bubby_geo/src_scatter').inputConnectors[0].connect(op('/project1/Bubby/bubby_geo/base_sphere'))"
)
```

### 降级方案：REST API（仅在 MCP 工具不可用时）

项目已安装 `touchdesigner-mcp`，位于 `tools/touchdesigner-mcp-td/`。
TD 端通过 WebServer DAT 暴露 REST API，默认端口 `9981`。

```python
import urllib.request, json
TD_URL = 'http://127.0.0.1:9981'

def td_exec(script: str) -> dict:
    req = urllib.request.Request(
        f'{TD_URL}/api/td/server/exec',
        data=json.dumps({'script': script}).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req) as res:
        return json.loads(res.read().decode())
```

---

## 2. TD 2026 踩坑速查表 (血泪教训)

> [!CAUTION]
> 以下所有坑都是在实际开发中用 TD 2026 真机验证过的。违反其中任何一条都会导致黑屏或报错。

| # | 坑 | 典型报错 | 正确做法 |
|---|---|---------|---------|
| 1 | `noiseSOP.par.roughness` | AttributeError | **不存在！** 只用 `amp`, `tz.expr` |
| 2 | `noiseSOP.par.harmonics` | AttributeError | **不存在！** noiseTOP 也一样 |
| 3 | `geo.par.material = 'mat_name'` (相对路径) | 渲染黑屏 | 必须用 **绝对路径** `mat.path` |
| 4 | `srcblend='srcalpha'` + constantMAT | 全黑 | constantMAT alpha=0，用 `srcblend='one'` |
| 5 | `geo.par.sx.expr = "sin(...)"` | NameError: sin | TD 表达式无 `math`/`sin`，用三角波算术 |
| 6 | `geo.create('sphere', ...)` (字符串类型) | Unknown operator type | 必须传全局类 `sphereSOP`、`noiseSOP` 等 |
| 7 | 代码创建 `geometryCOMP` | 自带默认 `torus1` | 创建后**立刻遍历删除**非目标子节点 |
| 8 | SphereSOP 直接用于粒子点云 | 两极过曝、中心暗 | 用 `scatterSOP` 均匀散布 |
| 9 | `par.page.name` 过滤 | AttributeError (None) | 先判断 `p.page is not None` |
| 10 | `td.scatterSOP` (错误模块前缀) | module 'td' has no attribute | 全局类直接用 `scatterSOP`，不加 `td.` |
| 11 | `td.sopToCHOP` | module 'td' has no attribute | **不存在！** 直接用 `instancesop` 连 SOP |
| 12 | `compositeTOP.par.operand` | 需要正确字符串 | 已验证值：`'add'`, `'over'`, `'multiply'` 等 |

### TD 2026 已验证的正确全局 SOP 类名

```python
# 这些可以直接在 geo.create() 中使用
sphereSOP    # 球体
noiseSOP     # 噪波变形
scatterSOP   # 均匀散布点
boxSOP       # 长方体
convertSOP   # 类型转换
```

### geometryCOMP Instancing 已验证参数名

```python
pgeo.par.instancing = 1         # 开启实例化 (布尔)
pgeo.par.instancesop = sop.path # SOP 数据源 (绝对路径!)
pgeo.par.instancetx = 'tx'      # X 位置通道
pgeo.par.instancety = 'ty'      # Y 位置通道
pgeo.par.instancetz = 'tz'      # Z 位置通道
pgeo.par.instancesx = 'sx'      # X 缩放通道 (可选)
pgeo.par.instancesy = 'sy'      # Y 缩放通道 (可选)
pgeo.par.instancesz = 'sz'      # Z 缩放通道 (可选)
pgeo.par.instancerx = 'rx'      # X 旋转通道 (可选)
```

### constantMAT 加法混合正确设置

```python
mat.par.colorr = 0.015   # 极低值，让加法叠加自然累积
mat.par.colorg = 0.08
mat.par.colorb = 0.25
mat.par.blending = 1
mat.par.blendop = 'add'
mat.par.srcblend = 'one'   # ← 关键！不是 srcalpha！
mat.par.destblend = 'one'
mat.par.depthwriting = 0   # 不写深度 (粒子可透视)
mat.par.depthtest = 1      # 测试深度 (保留前后关系)
```

---

## 3. GPU Instancing 粒子球 — 完整管线

这是我们验证过的最稳定的粒子球实现方案。

### 架构图

```
[bubby_geo geometryCOMP]
  ├── sphereSOP (base_sphere)    ← 球面模板
  ├── scatterSOP (src_scatter)   ← 均匀散布 2000 点
  └── noiseSOP (src_noise)       ← Perlin 流体扰动
              ↓ (instancesop 绝对路径)
[particle_geo geometryCOMP]
  ├── boxSOP (voxel, size=0.022) ← 每个粒子形状
  └── constantMAT (additive blue) ← 加法混合材质
              ↓
  [bubby_render renderTOP]
              ↓
  [glow_blur blurTOP] → [glow_comp compositeTOP(add)]
              ↓
  [final_output overTOP + text]
```

### 完整 Python 构建脚本

```python
# 通过 td_exec() 发送以下脚本到 TD
script = """
from td import op
import td

bubby = op('/project1/Bubby')
geo = op('/project1/Bubby/bubby_geo')

# --- 清理旧节点 ---
keep = {'bubby_cam','bubby_light','bubby_render','final_output','thought_text','bubby_geo'}
for c in list(bubby.children):
    if c.name not in keep:
        try: c.destroy()
        except: pass
for c in list(geo.children):
    try: c.destroy()
    except: pass

# --- Step 1: SOP 点云 ---
sphere = geo.create(sphereSOP, 'base_sphere')
sphere.par.radx = 1.2; sphere.par.rady = 1.2; sphere.par.radz = 1.2
sphere.par.rows = 30; sphere.par.cols = 30

scatter = geo.create(scatterSOP, 'src_scatter')
scatter.inputConnectors[0].connect(sphere)
try: scatter.par.npts = 2000
except:
    try: scatter.par.number = 2000
    except: pass

noise = geo.create(noiseSOP, 'src_noise')
noise.inputConnectors[0].connect(scatter)
noise.par.amp = 0.08
noise.par.tz.expr = 'absTime.seconds * 0.2'
noise.display = True; noise.render = True

# --- Step 2: 实例容器 ---
pgeo = bubby.create(geometryCOMP, 'particle_geo')

# 删除默认 torus（TD 自动创建的！）
for c in list(pgeo.children):
    try: c.destroy()
    except: pass

# 微型方块
voxel = pgeo.create(boxSOP, 'voxel')
voxel.par.sizex = 0.022; voxel.par.sizey = 0.022; voxel.par.sizez = 0.022
voxel.display = True; voxel.render = True

# 加法混合材质
pmat = pgeo.create(constantMAT, 'voxel_mat')
pmat.par.colorr = 0.015; pmat.par.colorg = 0.08; pmat.par.colorb = 0.25
pmat.par.blending = 1; pmat.par.blendop = 'add'
pmat.par.srcblend = 'one'; pmat.par.destblend = 'one'
pmat.par.depthwriting = 0; pmat.par.depthtest = 1
pgeo.par.material = pmat.path  # 绝对路径!

# 实例化绑定
pgeo.par.instancing = 1
pgeo.par.instancesop = noise.path  # 绝对路径!
pgeo.par.instancetx = 'tx'
pgeo.par.instancety = 'ty'
pgeo.par.instancetz = 'tz'

# --- Step 3: 呼吸 (4 秒三角波) ---
breath = "abs((absTime.frame % 240) / 120.0 - 1.0) * 0.05 + 0.975"
pgeo.par.sx.expr = breath
pgeo.par.sy.expr = breath
pgeo.par.sz.expr = breath

# --- Step 4: 渲染 + 辉光 ---
render = op('/project1/Bubby/bubby_render')
render.par.geometry = pgeo.name
render.par.camera = 'bubby_cam'
render.par.lights = ''
render.par.bgcolorr = 0; render.par.bgcolorg = 0
render.par.bgcolorb = 0.015; render.par.bgcolora = 1

glow_b = bubby.create(td.blurTOP, 'glow_blur')
glow_b.inputConnectors[0].connect(render)
glow_b.par.size = 0.055

glow_c = bubby.create(td.compositeTOP, 'glow_comp')
glow_c.inputConnectors[0].connect(render)
glow_c.inputConnectors[1].connect(glow_b)
glow_c.par.operand = 'add'

final = op('/project1/Bubby/final_output')
text = op('/project1/Bubby/thought_text')
if text:
    text.par.bgalpha = 0.0; text.par.fontalpha = 1.0
if final:
    final.inputConnectors[0].connect(text)
    final.inputConnectors[1].connect(glow_c)
    final.openViewer(unique=False, borders=True)

print('PARTICLE SPHERE BUILT SUCCESSFULLY!')
"""
```

---

## 4. 高级：GLSL 粒子系统 (参考资源)

如果需要超越 Instancing 的百万级粒子效果，参考以下 GitHub 库：

### Boid 群集粒子 — `heysoos/td_swarm_particles`
- **核心文件**: `particle_shader.glsl` — 粒子间吸引/排斥力的 GLSL 计算着色器
- **原理**: 在 GLSL TOP 中以 Compute Shader 模式运行，每帧更新粒子位置/速度纹理
- **用法**: 直接打开 `ParticleComputeShader.GUI.toe`，右键 particles 容器 → View

### 流场粒子 — `vjasterix/TD-Particle-Flowfields`
- **原理**: 粒子运动完全在 GPU 纹理中计算和存储
- **核心技术**: 
  - Noise TOP 生成 3D 流场纹理
  - 每个像素的 RGB = 该位置的力场方向
  - GLSL TOP 读取流场 + 当前位置纹理 → 输出下一帧位置纹理
  - Feedback TOP 形成循环

### GLSL 粒子通用架构

```
[Noise TOP] → 流场/力场数据
      ↓
[GLSL TOP (Compute)] ← [Feedback TOP] ← 自身输出
      ↓                                    ↑
  位置纹理 (每像素 = 一个粒子的 XYZ)        │
      ↓                                    │
  [实例化渲染] ──────────────────────────────┘
```

---

## 5. 情感联动接口 (Bubby 专用)

通过现有的 `scripts/st_td_bridge.py` 将 Bubby 情感映射到粒子参数：

```python
# 在 st_td_bridge.py 的 emotion_update 中
emotion_map = {
    'calm':    {'amp': 0.04, 'speed': 0.1, 'r': 0.01, 'g': 0.06, 'b': 0.2},
    'excited': {'amp': 0.15, 'speed': 0.5, 'r': 0.03, 'g': 0.1,  'b': 0.4},
    'sad':     {'amp': 0.02, 'speed': 0.05,'r': 0.02, 'g': 0.03, 'b': 0.15},
}

def apply_emotion(emotion: str):
    cfg = emotion_map.get(emotion, emotion_map['calm'])
    td_exec(f"""
        noise = op('/project1/Bubby/bubby_geo/src_noise')
        noise.par.amp = {cfg['amp']}
        mat = op('/project1/Bubby/particle_geo/voxel_mat')
        mat.par.colorr = {cfg['r']}
        mat.par.colorg = {cfg['g']}
        mat.par.colorb = {cfg['b']}
    """)
```

---

## 6. 诊断检查清单

遇到黑屏或形状异常时，按此顺序排查：

```python
# 发送此诊断脚本
diag = """
render = op('/project1/Bubby/bubby_render')
pgeo = op('/project1/Bubby/particle_geo')
geo = op('/project1/Bubby/bubby_geo')

print('--- DIAGNOSTICS ---')
print('render geometry:', render.par.geometry.val)
print('render errors:', render.errors())
print('pgeo instancing:', pgeo.par.instancing.val)
print('pgeo instancesop:', pgeo.par.instancesop.val)
print('pgeo material:', pgeo.par.material.val)
print('pgeo errors:', pgeo.errors())
print('pgeo children:', [(c.name, c.OPType) for c in pgeo.children])
print('geo children:', [(c.name, c.OPType) for c in geo.children])
"""
```

| 症状 | 原因 | 修复 |
|------|------|------|
| 全黑 | material 路径错 | `pgeo.par.material = mat.path` (绝对) |
| 全黑 | srcblend=srcalpha | 改为 `srcblend='one'` |
| 椭圆/甜甜圈 | 默认 torus1 未删 | 创建后立刻遍历删除 |
| 两极过亮 | SphereSOP 极点密集 | 换用 scatterSOP |
| 不动 | noiseSOP tz 表达式没设 | `noise.par.tz.expr = 'absTime.seconds * 0.2'` |
| 报错 sin | TD 表达式无 math | 用 absTime.frame 三角波 |
