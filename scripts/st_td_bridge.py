import asyncio
import json
import logging
import re
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import websockets

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("st_td_bridge")

# FastAPI App for receiving Webhooks from ST
app = FastAPI(title="SillyTavern to TouchDesigner Bridge")

# Enable CORS for local ST access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
TD_WEBSOCKET_URL = "ws://localhost:9980"  # Assuming TD websocketDAT is listening here
READING_SPEED_CHARS_PER_SEC = 10.0 # Characters read per second

# Emotion Dictionary (Can be expanded)
# Format: "regex_pattern": (mood_intensity(0-1), base_color(r,g,b))
EMOTION_MAP = {
    r"愤怒|生气|失控|质问|咬牙|冷笑|瞪": (0.9, [0.8, 0.1, 0.1]),   # Angry - Red
    r"哭|泣|泪|哽咽|悲伤|痛苦": (0.8, [0.1, 0.3, 0.8]),          # Sad - Blue
    r"害怕|恐惧|颤抖|退缩|紧张": (0.7, [0.5, 0.1, 0.5]),         # Fear - Purple
    r"开心|愉快|笑|喜悦|活泼|轻快": (0.6, [0.9, 0.8, 0.2]),         # Happy - Yellow
    r"挑逗|魅惑|耳语|戏谑|贴近": (0.6, [0.8, 0.2, 0.6]),          # Flirty - Magenta
    r"脸红|害羞|局促|支吾": (0.5, [0.9, 0.4, 0.4]),              # Shy - Soft Red
    r"温柔|轻抚|安抚|微笑|轻声": (0.3, [0.9, 0.6, 0.7]),          # Tender - Pink
    r"冷淡|漠然|面无表情|淡淡": (0.15, [0.7, 0.7, 0.8]),         # Cold - Grayish blue
    r"平静|静静|看着|停顿": (0.1, [0.4, 0.6, 0.8]),           # Calm - Muted Blue
    r"沉默|良久|无言": (0.05, [0.3, 0.3, 0.3]),               # Silent - Dark Gray
}
DEFAULT_EMOTION = (0.3, [0.5, 0.7, 0.9]) # Default calm cyan state

def clean_st_text(text: str) -> str:
    """
    Cleans up SillyTavern extended markdown, metadata, drafts, and HTML tags,
    keeping only the actual story content that the user sees, to prevent
    the bridge from reacting to hidden prompt details.
    """
    # 1. If there's a <content> block, extract only that part
    content_match = re.search(r"<content>(.*?)</content>", text, flags=re.DOTALL | re.IGNORECASE)
    if content_match:
        text = content_match.group(1)

    # 2. Strip HTML comments (e.g., <!-- draft: ... -->)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    
    # 3. Strip prologue and other common metadata blocks
    text = re.sub(r"<prologue>.*?</prologue>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<aurora_time>.*?</aurora_time>", "", text, flags=re.DOTALL | re.IGNORECASE)
    
    # 4. Strip specific tag markers like <FH_001>, <li> but keep their inner text
    text = re.sub(r"</?(?:FH_\d+|li|bgm|snow|details|summary|p|br|span)[^>]*>", "", text, flags=re.IGNORECASE)
    
    # 5. Strip bracketed meta text e.g. [finire]
    text = re.sub(r"\[.*?\]", "", text)
    
    return text.strip()

def parse_emotions_time_series(raw_text: str) -> list[dict]:
    """
    Cleans raw ST text, splits it into sentences, analyzes emotion per sentence,
    and assigns an estimated display duration.
    """
    # First, run the text through our cleaner to remove drafts and metadata
    text = clean_st_text(raw_text)
    
    if not text:
        return []
        
    # Split by common punctuation marks that indicate a pause
    sentences = re.split(r'([。！？…\n]+)', text)
    
    time_series = []
    
    # Re-combine split strings with their punctuation
    combined_sentences = []
    for i in range(0, len(sentences)-1, 2):
        combined_sentences.append(sentences[i] + sentences[i+1])
    if len(sentences) % 2 != 0 and sentences[-1].strip():
        combined_sentences.append(sentences[-1])
        
    if not combined_sentences:
        combined_sentences = [text]

    for sent in combined_sentences:
        clean_sent = sent.strip()
        if not clean_sent:
            continue
            
        current_mood, current_color = DEFAULT_EMOTION
        
        # Determine emotion for this sentence
        for pattern, (mood, color) in EMOTION_MAP.items():
            if re.search(pattern, clean_sent):
                current_mood = mood
                current_color = color
                break
                
        # Estimate duration based on length
        duration_sec = max(2.0, len(clean_sent) / READING_SPEED_CHARS_PER_SEC)
        
        # Extract a short snippet for the thought visual
        thought_snippet = clean_sent[:30] + "..." if len(clean_sent) > 30 else clean_sent
        
        time_series.append({
            "duration": duration_sec,
            "data": {
                "mood_intensity": current_mood,
                "base_color": current_color,
                "thought_text": thought_snippet,
                "trigger_event": False
            }
        })
        
    return time_series

import urllib.request
import urllib.error

# TouchDesigner MCP WebServer URL
TD_API_URL = "http://127.0.0.1:9981/api/td/server/exec"

current_stream_task = None

async def stream_to_td(time_series: list[dict]):
    """
    Connects to TD via the WebServer MCP API and streams the discrete emotion states 
    over time to simulate a continuous performance.
    """
    try:
        logger.info("Starting emotion stream via TD HTTP API!")
        
        for item in time_series:
            duration = item["duration"]
            payload = item["data"]
            
            logger.info(f"Sending to TD [Wait {duration:.1f}s]: Mood={payload['mood_intensity']:.2f}, Text='{payload['thought_text']}'")
            
            # Construct a Python script to execute remotely inside TD
            mood = payload['mood_intensity']
            c = payload['base_color']
            txt = payload['thought_text'].replace('"', '\\"').replace('\n', ' ')
            
            td_script = f"""
try:
    from td import op, absTime, run
    noise = op('/project1/Bubby/bubby_geo/bubby_noise')
    mat = op('/project1/Bubby/bubby_mat')
    text = op('/project1/Bubby/thought_text')
    drop = op('/project1/Bubby/text_drop')
    
    if noise:
        noise.par.amp = {0.05 + mood * 0.15}
    if mat:
        mat.par.basecolorr = {c[0]}
        mat.par.basecolorg = {c[1]}
        mat.par.basecolorb = {c[2]}
    if text and drop:
        text.par.text = "{txt}"
        text.par.fontalpha = 1.0
        
        drop.store('start_time', absTime.seconds)
        drop.par.ty.expr = "0.2 - (absTime.seconds - me.fetch('start_time', absTime.seconds)) * 0.15"
        
        run('op("/project1/Bubby/thought_text").par.fontalpha = 0.0', delayFrames=90)
except Exception as e:
    pass
"""
            # Send the script via HTTP POST
            data = json.dumps({'script': td_script}).encode('utf-8')
            req = urllib.request.Request(TD_API_URL, data=data, headers={'Content-Type': 'application/json'})
            try:
                with urllib.request.urlopen(req) as response:
                    pass # Success
            except urllib.error.URLError as e:
                logger.error(f"Could not connect to TD at {TD_API_URL}. Is the TD WebServer running? Error: {e}")
                
            # Wait for the estimated reading duration of this sentence
            await asyncio.sleep(duration)
            
        logger.info("Finished streaming emotion sequence to TD.")
        
    except asyncio.CancelledError:
        logger.info("Emotion stream interrupted by new message.")
        raise
    except Exception as e:
         logger.error(f"HTTP stream error: {e}")

@app.post("/webhook/st_message")
async def receive_st_message(request: Request):
    """
    Webhook endpoint for SillyTavern to hit via slash command or extension.
    Expected JSON: {"name": "CharacterName", "text": "The long reply text..."}
    """
    global current_stream_task
    
    try:
        data = await request.json()
        message_text = data.get("text", "")
        char_name = data.get("name", "Unknown")
        
        if not message_text:
            return {"status": "ignored", "reason": "empty text"}
            
        logger.info(f"Received message from ST ({char_name}): {len(message_text)} chars")
        
        # 1. Parse the long text into a sequence of emotions
        time_series = parse_emotions_time_series(message_text)
        
        if not time_series:
            return {"status": "ignored", "reason": "no valid sentences parsed"}
            
        # 2. Cancel previous streaming task if still active
        if current_stream_task and not current_stream_task.done():
            current_stream_task.cancel()
            
        # 3. Start new streaming task in background
        current_stream_task = asyncio.create_task(stream_to_td(time_series))
        
        return {
            "status": "success", 
            "message": f"Queued {len(time_series)} emotion states for streaming to TD."
        }
        
    except Exception as e:
        logger.error(f"Error processing ST webhook: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting ST to TD Bridge on port 8001...")
    logger.info(f"Expecting TD WebSocket Server at: {TD_WEBSOCKET_URL}")
    uvicorn.run(app, host="0.0.0.0", port=8001)
