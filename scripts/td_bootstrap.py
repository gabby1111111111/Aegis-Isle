import urllib.request, json
import sys

# The exact setup script code we want executed
setup_code = """
from td import op
root = op('/project1')

# Create an isolated baseCOMP to prevent clutter and validate smoothly
bubby = root.op('Bubby')
if not bubby:
    bubby = root.create(baseCOMP, 'Bubby')
    bubby.nodeX = 400
    bubby.nodeY = 0
    
# Clean existing nodes inside if recreating
for c in bubby.children:
    c.destroy()

mat = bubby.create(pbrMAT, 'bubby_mat')

geo = bubby.create(geometryCOMP, 'bubby_geo')
for c in geo.children:
    c.destroy()

sphere = geo.create(sphereSOP, 'bubby_body')
sphere.par.rows = 40
sphere.par.cols = 40

noise = geo.create(noiseSOP, 'bubby_noise')
noise.par.amp = 0.05
noise.par.tz.expr = 'absTime.seconds * 0.2'
noise.inputConnectors[0].connect(sphere)

noise.render = True
noise.display = True
geo.par.material = mat.path

cam = bubby.create(cameraCOMP, 'bubby_cam')
cam.par.tz = 5

light = bubby.create(lightCOMP, 'bubby_light')

render = bubby.create(renderTOP, 'bubby_render')
render.par.resolutionw = 1280
render.par.resolutionh = 720

text = bubby.create(textTOP, 'thought_text')
text.par.resolutionw = 1280
text.par.resolutionh = 720
text.par.fontsizex = 40
text.par.text = 'Awakening...'
text.par.fontalpha = 0.0 

drop = bubby.create(transformTOP, 'text_drop')
drop.inputConnectors[0].connect(text)

trail = bubby.create(feedbackTOP, 'thought_trail')

level = bubby.create(levelTOP, 'trail_fade')
level.par.opacity = 0.95

trail_move = bubby.create(transformTOP, 'trail_move')
trail_move.par.ty = 0.005

# Connect feedback loop
trail.inputConnectors[0].connect(drop)
level.inputConnectors[0].connect(trail)
trail_move.inputConnectors[0].connect(level)
trail.par.top = trail_move.name

comp = bubby.create(overTOP, 'trail_comp')
comp.inputConnectors[0].connect(drop)
comp.inputConnectors[1].connect(trail_move)

final_output = bubby.create(overTOP, 'final_output')
final_output.inputConnectors[0].connect(comp)
final_output.inputConnectors[1].connect(render)

# Automatically show the output viewer
final_output.viewer = True
"""

# Encode the script to base64 to avoid ALL string literal escaping nightmare loops!
import base64
setup_code_b64 = base64.b64encode(setup_code.encode('utf-8')).decode('utf-8')

bootstrap_script = f"""
try:
    import base64
    from td import op
    proj = op('/')
    tmp = proj.op('auto_installer')
    if tmp: 
        tmp.destroy()
    from td import textDAT
    tmp = proj.create(textDAT, 'auto_installer')
    tmp.text = base64.b64decode('{setup_code_b64}').decode('utf-8')
    tmp.run(delayFrames=1)
    print("Bootstrap planted!")
except Exception as e:
    import traceback
    print("Bootstrap error:", e, traceback.format_exc())
"""

url = 'http://127.0.0.1:9981/api/td/server/exec'
req = urllib.request.Request(url, data=json.dumps({'script': bootstrap_script}).encode('utf-8'), headers={'Content-Type': 'application/json'})
try:
    with urllib.request.urlopen(req) as res:
        print('Bootstrap Sent Successfully:', res.read().decode())
except Exception as e:
    print('Bootstrap Failed:', e)
