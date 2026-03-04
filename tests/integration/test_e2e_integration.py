"""
端到端集成测试 - 模拟 SillyTavern 请求
测试完整的状态管理功能
"""

import asyncio
import aiohttp
import json
from datetime import datetime

BASE_URL = "http://127.0.0.1:8001"
USER_ID = "test_user_e2e"


async def test_chat_completion(session, user_message: str, round_num: int):
    """测试聊天完成接口"""
    print(f"\n{'='*60}")
    print(f"🎮 轮次 {round_num}: {user_message}")
    print(f"{'='*60}")
    
    url = f"{BASE_URL}/v1/chat/completions"
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [
            {"role": "user", "content": user_message}
        ],
        "stream": False,
        "user": USER_ID
    }
    
    try:
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                ai_response = data["choices"][0]["message"]["content"]
                print(f"✅ AI 回复: {ai_response[:200]}...")
                print(f"📊 Token 使用: {data.get('usage', {})}")
                return ai_response
            else:
                error_text = await response.text()
                print(f"❌ 请求失败 ({response.status}): {error_text}")
                return None
    except Exception as e:
        print(f"❌ 请求异常: {e}")
        return None


async def check_user_state(session):
    """检查用户状态"""
    print(f"\n📋 检查用户状态...")
    
    url = f"{BASE_URL}/v1/state/{USER_ID}"
    
    try:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                print(f"✅ 状态版本: {data.get('version', 'N/A')}")
                print(f"📦 表格数量: {len(data.get('sheets_summary', {}))}")
                
                for sheet_name, info in data.get('sheets_summary', {}).items():
                    print(f"   - {info['name']}: {info['row_count']} 行")
                
                return data
            else:
                print(f"❌ 获取状态失败 ({response.status})")
                return None
    except Exception as e:
        print(f"❌ 获取状态异常: {e}")
        return None


async def check_snapshots(session):
    """检查快照列表"""
    print(f"\n📸 检查快照列表...")
    
    url = f"{BASE_URL}/v1/state/{USER_ID}/snapshots"
    
    try:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                snapshots = data.get('snapshots', [])
                print(f"✅ 快照数量: {len(snapshots)}")
                
                for i, snap in enumerate(snapshots[:3], 1):
                    print(f"   {i}. {snap['snapshot_id']}")
                    print(f"      时间: {snap['timestamp']}")
                    print(f"      摘要: {snap['change_summary']}")
                
                return snapshots
            else:
                print(f"❌ 获取快照失败 ({response.status})")
                return []
    except Exception as e:
        print(f"❌ 获取快照异常: {e}")
        return []


async def test_rollback(session, snapshot_id: str):
    """测试回滚功能"""
    print(f"\n⏪ 测试回滚到快照: {snapshot_id}")
    
    url = f"{BASE_URL}/v1/state/{USER_ID}/rollback"
    payload = {"snapshot_id": snapshot_id}
    
    try:
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                print(f"✅ 回滚成功!")
                print(f"   恢复版本: {data.get('restored_version', 'N/A')}")
                return True
            else:
                error_text = await response.text()
                print(f"❌ 回滚失败 ({response.status}): {error_text}")
                return False
    except Exception as e:
        print(f"❌ 回滚异常: {e}")
        return False


async def main():
    """主测试流程"""
    print("🚀 开始端到端集成测试")
    print(f"⏰ 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 目标服务器: {BASE_URL}")
    print(f"👤 测试用户: {USER_ID}")
    
    async with aiohttp.ClientSession() as session:
        # 测试场景
        test_scenarios = [
            "我在新手村买了一把木剑",
            "村长送了我3瓶红色药水",
            "我喝了一瓶红色药水",
            "我又捡到了一个蓝色药水",
            "查看我的背包"
        ]
        
        # 轮次 1-5: 正常对话
        for i, message in enumerate(test_scenarios, 1):
            response = await test_chat_completion(session, message, i)
            if response:
                await asyncio.sleep(1)  # 等待后台任务完成
        
        # 检查状态
        print("\n" + "="*60)
        print("📊 第一次状态检查")
        print("="*60)
        state = await check_user_state(session)
        
        # 检查快照
        snapshots = await check_snapshots(session)
        
        # 测试回滚 (如果有快照)
        if len(snapshots) >= 2:
            print("\n" + "="*60)
            print("🔄 测试快照回滚功能")
            print("="*60)
            
            # 回滚到第二个快照
            second_snapshot = snapshots[1]['snapshot_id']
            success = await test_rollback(session, second_snapshot)
            
            if success:
                # 再次检查状态
                print("\n" + "="*60)
                print("📊 回滚后状态检查")
                print("="*60)
                await check_user_state(session)
        
        # 最终总结
        print("\n" + "="*60)
        print("✅ 测试完成!")
        print("="*60)
        print("\n📝 测试总结:")
        print(f"   - 对话轮次: {len(test_scenarios)}")
        print(f"   - 快照数量: {len(snapshots)}")
        print(f"   - 状态管理: {'✅ 正常' if state else '❌ 失败'}")
        print(f"   - 快照系统: {'✅ 正常' if snapshots else '❌ 失败'}")
        
        print("\n🔗 验证链接:")
        print(f"   - 状态: {BASE_URL}/v1/state/{USER_ID}")
        print(f"   - 快照: {BASE_URL}/v1/state/{USER_ID}/snapshots")
        print(f"   - API 文档: {BASE_URL}/docs")


if __name__ == "__main__":
    asyncio.run(main())
