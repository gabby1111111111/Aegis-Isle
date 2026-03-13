#!/usr/bin/env python3
"""
📱 Bubby Mobile Sensor v2 (Termux 版)
让 Char 感知你手机上的一切 — 三通道联动版

三通道架构:
  ① Aegis-Isle /v1/diary/event     → 长期记忆存储
  ② ST Companion-Link /inject      → 角色 in-character 回复（酒馆内）
  ③ ntfy 推送                      → 手机弹窗通知（酒馆外）

使用方法:
  1. 手机上打开 Termux
  2. pkg install termux-api python
  3. pip install requests
  4. 修改下面的 AEGIS_HOST / ST_HOST 为你电脑的局域网 IP
  5. python bubby_mobile_sensor.py
"""

import subprocess
import requests
import time
import json
import random
import logging
from datetime import datetime

# ============================================
# 配置区 — 改这里 ↓
# ============================================

# 你电脑的局域网 IP
PC_IP = "192.168.1.100"         # ← 改成你电脑的 IP

# 通道 ① Aegis-Isle（记忆存储）
AEGIS_PORT = 8001
AEGIS_URL = f"http://{PC_IP}:{AEGIS_PORT}/v1/diary/event"

# 通道 ② SillyTavern Companion-Link（角色回复）
ST_PORT = 8000
ST_INJECT_URL = f"http://{PC_IP}:{ST_PORT}/api/plugins/companion-link/inject"

# 通道 ③ ntfy（手机弹窗推送）
NTFY_TOPIC = "gabby-ring"  # ← 你的 ntfy 频道名
NTFY_SERVER = "https://ntfy.sh"              # 公共服务器，也可以改成自建的

# 检查间隔（秒）
CHECK_INTERVAL = 30

# 连续刷多少分钟后触发提醒
NAG_THRESHOLD_MINUTES = 30

# 深夜不睡觉提醒
LATE_NIGHT_HOUR = 1

# ============================================
# App 识别表
# ============================================
APP_NAMES = {
    "com.xingin.xhs": "小红书",
    "com.ss.android.ugc.aweme": "抖音",
    "com.ss.android.article.news": "今日头条",
    "com.tencent.mm": "微信",
    "com.tencent.mobileqq": "QQ",
    "tv.danmaku.bili": "哔哩哔哩",
    "com.miHoYo.Nap": "绝区零",
    "com.mihoyo.hyperion": "米游社",
    "com.miHoYo.GenshinImpact": "原神",
    "com.netease.cloudmusic": "网易云音乐",
    "com.zhihu.android": "知乎",
    "com.taobao.taobao": "淘宝",
    "com.tencent.tmgp.sgame": "王者荣耀",
    "com.sankuai.meituan": "美团",
    "me.ele": "饿了么",
}

APP_CATEGORIES = {
    "game": ["com.miHoYo.Nap", "com.miHoYo.GenshinImpact", "com.tencent.tmgp.sgame"],
    "video": ["tv.danmaku.bili", "com.ss.android.ugc.aweme"],
    "social": ["com.xingin.xhs"],
    "messaging": ["com.tencent.mm", "com.tencent.mobileqq"],
}

IGNORE_PACKAGES = {
    "com.android.launcher", "com.android.systemui",
    "com.android.settings", "com.android.vending", "com.termux",
}

# ============================================
# 日志
# ============================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [📱] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("sensor")

# ============================================
# 工具函数
# ============================================

def get_current_app():
    try:
        result = subprocess.run(["dumpsys", "activity", "recents"], capture_output=True, text=True, timeout=5)
        for line in result.stdout.split("\n"):
            if "realActivity=" in line:
                return line.split("realActivity=")[1].split("/")[0].strip()
    except Exception:
        pass
    return None

def get_friendly_name(pkg):
    return APP_NAMES.get(pkg, pkg.split(".")[-1] if pkg else "未知")

def get_category(pkg):
    for cat, pkgs in APP_CATEGORIES.items():
        if pkg in pkgs:
            return cat
    return "app"

def get_time_context():
    h = datetime.now().hour
    if h < 6: return "凌晨"
    elif h < 9: return "早上"
    elif h < 12: return "上午"
    elif h < 14: return "午后"
    elif h < 18: return "下午"
    elif h < 22: return "晚上"
    else: return "深夜"

# ============================================
# 通道 ① Aegis-Isle 记忆
# ============================================
def post_aegis(pkg, name, duration=0):
    cat = get_category(pkg)
    tc = get_time_context()
    verb = "玩" if cat == "game" else "刷" if cat in ["video", "social"] else "用"
    comment = f"user {tc}在手机上{verb}{name}"
    if duration > 60:
        comment += f"，持续了约{duration // 60}分钟"

    try:
        requests.post(AEGIS_URL, json={
            "source": "browsing", "action": f"phone_{cat}",
            "title": f"{tc}在手机上用{name}",
            "tags": [name, cat, tc, "手机"],
            "url": "", "platform": f"phone_{cat}", "comment": comment
        }, timeout=3)
        logger.info(f"① Aegis: {comment}")
    except Exception as e:
        logger.warning(f"① Aegis 失败: {e}")

# ============================================
# 通道 ② ST Companion-Link 角色回复
# ============================================
def post_st_inject(pkg, name, minutes, trigger_type="nag"):
    tc = get_time_context()
    cat = get_category(pkg)

    if trigger_type == "nag":
        text = f"（user 已经连续{('刷' if cat in ['video','social'] else '玩' if cat == 'game' else '用')}{name}{minutes}分钟了"
        if datetime.now().hour >= 22 or datetime.now().hour < 6:
            text += f"，现在是{tc}"
        text += "。请以你的方式关心或提醒 user。）"
    elif trigger_type == "late_night":
        text = f"（现在是凌晨{datetime.now().hour}点，user 还在{('刷' if cat in ['video','social'] else '玩' if cat == 'game' else '用')}{name}没有睡觉。请以你的方式关心 user。）"
    elif trigger_type == "meal":
        text = f"（现在是{'中午' if datetime.now().hour < 14 else '傍晚'}饭点了，user 可能还没吃饭。请关心一下 user 的饮食。）"
    else:
        text = f"（user 在手机上{name}。）"

    try:
        requests.post(ST_INJECT_URL, json={
            "action": f"phone_{trigger_type}",
            "formatted_text": text,
            "note": {"title": f"手机{'提醒' if trigger_type == 'nag' else '关心'}", "tags": [name, tc]},
            "timestamp": datetime.now().isoformat()
        }, timeout=3)
        logger.info(f"② ST inject: {text[:60]}...")
    except Exception as e:
        logger.warning(f"② ST inject 失败: {e}")

# ============================================
# 通道 ③ ntfy 手机弹窗
# ============================================
def post_ntfy(title, message):
    try:
        requests.post(
            f"{NTFY_SERVER}/{NTFY_TOPIC}",
            data=message.encode("utf-8"),
            headers={"Title": title, "Tags": "bell"},
            timeout=3
        )
        logger.info(f"③ ntfy: {title}")
    except Exception as e:
        logger.warning(f"③ ntfy 失败: {e}")

# ============================================
# 触发逻辑
# ============================================
def trigger_nag(pkg, name, minutes):
    """连续刷太久 → 三通道提醒"""
    post_st_inject(pkg, name, minutes, "nag")
    post_ntfy("💬 角色想跟你说话", f"你已经{get_category(pkg) == 'game' and '玩' or '刷'}{name} {minutes}分钟了，去酒馆看看~")

def trigger_late_night(pkg, name):
    """深夜还在玩 → 三通道提醒"""
    post_st_inject(pkg, name, 0, "late_night")
    post_ntfy("🌙 深夜关心", f"凌晨{datetime.now().hour}点了还在{('玩' if get_category(pkg) == 'game' else '刷')}{name}？去酒馆看看角色说了什么")

def trigger_meal():
    """饭点提醒"""
    post_st_inject("", "吃饭", 0, "meal")
    post_ntfy("🍱 饭点关心", "该吃饭了！去酒馆看看角色说了什么~")

# ============================================
# 主循环
# ============================================
def main():
    logger.info(f"🐰 Bubby Mobile Sensor v2 启动！")
    logger.info(f"① Aegis: {AEGIS_URL}")
    logger.info(f"② ST:    {ST_INJECT_URL}")
    logger.info(f"③ ntfy:  {NTFY_SERVER}/{NTFY_TOPIC}")
    logger.info(f"按 Ctrl+C 停止\n")

    last_app = None
    app_start_time = time.time()
    last_nag_time = 0
    last_meal_hour = -1

    while True:
        try:
            current_app = get_current_app()
            now = time.time()
            hour = datetime.now().hour

            if current_app and current_app not in IGNORE_PACKAGES:
                name = get_friendly_name(current_app)

                if current_app != last_app:
                    # App 切换 → 上报旧 App 到 Aegis
                    if last_app and last_app not in IGNORE_PACKAGES:
                        duration = int(now - app_start_time)
                        if duration > 60:
                            post_aegis(last_app, get_friendly_name(last_app), duration)

                    last_app = current_app
                    app_start_time = now
                    logger.info(f"📱 {name}")
                else:
                    # 同一个 App → 检查是否该提醒
                    mins = int((now - app_start_time) / 60)
                    if mins >= NAG_THRESHOLD_MINUTES and (now - last_nag_time) > 600:
                        trigger_nag(current_app, name, mins)
                        last_nag_time = now

            # 深夜检测
            if LATE_NIGHT_HOUR <= hour < 6 and current_app and current_app not in IGNORE_PACKAGES:
                if (now - last_nag_time) > 1800:
                    trigger_late_night(current_app, get_friendly_name(current_app))
                    last_nag_time = now

            # 饭点检测
            if hour in [11, 12, 17, 18] and hour != last_meal_hour:
                last_meal_hour = hour
                trigger_meal()

            time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            logger.info("\n🐰 Sensor 已停止。")
            break
        except Exception as e:
            logger.error(f"异常: {e}")
            time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
