import sys
import os
import asyncio
import logging

sys.path.append(r"E:\Aegis_Isle\AegisIsle_cc_ver\Aegis-Isle\scripts")
import st_td_bridge

# Make logging visible
st_td_bridge.logger.setLevel(logging.INFO)

async def main():
    test_text = """<content>
    Bubby 微微地颤抖着，突然咬牙质问了起来！<!-- draft: angry test --> <FH_001>
    “你们为什么要这样对我？”他瞪大了眼睛。
    片刻之后，他沉默了，只是静静地看着眼前的虚空。
    最后，他破涕为笑，脸上泛起红晕，轻声说道：“谢谢你还在……”
    </content>"""
    
    print("==== 1. Testing parser ====")
    series = st_td_bridge.parse_emotions_time_series(test_text)
    print(f"Parsed {len(series)} emotion segments.")
    for s in series:
        print(f"  [{s['duration']:.1f}s] Mood: {s['data']['mood_intensity']}, Color: {s['data']['base_color']}, Text: {s['data']['thought_text']}")
        
    print("\n==== 2. Streaming to TD ====")
    # Temporarily speed up testing by reducing duration
    for s in series:
        s['duration'] = 1.0
        
    await st_td_bridge.stream_to_td(series)
    print("Done!")

if __name__ == "__main__":
    asyncio.run(main())
