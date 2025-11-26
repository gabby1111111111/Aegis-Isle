#!/usr/bin/env python3
"""
测试审计日志功能
"""

import sys
sys.path.insert(0, 'src')

from aegis_isle.core.logging import audit_logger

def test_audit_logging():
    """测试审计日志"""

    print("=" * 50)
    print("测试审计日志功能")
    print("=" * 50)

    # 测试认证日志
    print("\n1. 测试认证日志...")
    audit_logger.log_authentication(
        action="login_success",
        username="testuser",
        outcome="success",
        ip_address="127.0.0.1",
        user_agent="Test Client"
    )
    print("✅ 认证日志已记录")

    # 测试授权日志
    print("\n2. 测试授权日志...")
    audit_logger.log_authorization(
        action="access_resource",
        user_id="user123",
        username="testuser",
        resource="/api/v1/documents",
        outcome="success",
        required_permissions=["read", "write"]
    )
    print("✅ 授权日志已记录")

    # 测试数据访问日志
    print("\n3. 测试数据访问日志...")
    audit_logger.log_data_access(
        action="query_documents",
        user_id="user123",
        username="testuser",
        resource="documents",
        query="test query"
    )
    print("✅ 数据访问日志已记录")

    # 测试API访问日志
    print("\n4. 测试API访问日志...")
    audit_logger.log_api_access(
        method="POST",
        endpoint="/api/v1/documents/upload",
        user_id="user123",
        username="testuser",
        ip_address="127.0.0.1",
        status_code=200,
        response_time_ms=125.5,
        request_id="req-12345"
    )
    print("✅ API访问日志已记录")

    # 测试安全事件日志
    print("\n5. 测试安全事件日志...")
    audit_logger.log_security_event(
        action="suspicious_activity_detected",
        level="warning",
        ip_address="192.168.1.100",
        threat_type="brute_force_attempt"
    )
    print("✅ 安全事件日志已记录")

    print("\n" + "=" * 50)
    print("✅ 所有审计日志测试完成！")
    print("=" * 50)
    print("\n📁 查看日志文件:")
    print("   logs/audit/audit_*.jsonl")
    print("\n💡 提示: 使用 jq 查看格式化的JSON:")
    print("   cat logs/audit/audit_*.jsonl | jq")

if __name__ == "__main__":
    test_audit_logging()
