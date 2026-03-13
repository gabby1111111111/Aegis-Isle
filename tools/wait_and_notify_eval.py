import time
import subprocess
import requests

NTFY_TOPIC = "gabby-ring"
TEST_CMDS = [
    'pytest "tests/test_rag_real_data.py::TestRealDataDeepEval::test_dim3_contextual_relevancy[library_meet]" -v -s',
    'pytest "tests/test_rag_real_data.py::TestRealDataDeepEval::test_dim1_faithfulness[library_meet]" -v -s'
]

def is_ingestion_running():
    try:
        output = subprocess.check_output(
            'wmic process where "name=\'python.exe\'" get commandline', 
            shell=True, text=True
        )
        return "ingest_v2_chunks" in output
    except Exception as e:
        print(f"Error checking processes: {e}")
        return False

print("Waiting for ingest_v2_chunks to finish...")
while is_ingestion_running():
    time.sleep(30)

print("\nIngestion finished! Running library_meet evaluation...")

results_msg = ""
success_count = 0

for cmd in TEST_CMDS:
    metric_name = "Contextual Relevancy" if "dim3" in cmd else "Faithfulness"
    print(f"Running {metric_name}...")
    try:
        # Run the pytest command
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding="utf-8")
        out = result.stdout
        
        score = "Unknown"
        # Try to parse the DeepEval score from output. e.g. score=0.85
        import re
        m = re.search(r'score=([\d\.]+)', out)
        if m:
            score = m.group(1)
            
        if result.returncode == 0:
            results_msg += f"✅ {metric_name}: {score}\n"
            success_count += 1
        else:
            results_msg += f"❌ {metric_name} Failed (score={score})\n"
            
    except Exception as e:
        results_msg += f"❌ {metric_name} Error\n"

if success_count == len(TEST_CMDS):
    title = "🎉 Aegis-Isle RAG质量提升"
    msg = f"FAISS巨块重切分完毕！\nlibrary_meet场景测试通过:\n{results_msg}\n大功告成，晚安！"
else:
    title = "⚠️ Aegis-Isle 测试失败"
    msg = f"FAISS切分完毕，但测试未全绿:\n{results_msg}\n明早再看日志吧，晚安！"

print(f"Sending ntfy: {title}\n{msg}")
try:
    requests.post(
        f"https://ntfy.sh/{NTFY_TOPIC}",
        data=msg.encode('utf-8'),
        headers={
            "Title": title.encode('utf-8'),
            "Tags": "tada" if success_count == len(TEST_CMDS) else "warning"
        }
    )
    print("Notification sent successfully.")
except Exception as e:
    print(f"Failed to send notification: {e}")
