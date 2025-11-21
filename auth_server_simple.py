#!/usr/bin/env python
"""
AegisIsle 认证系统演示服务器 (简化版)
使用预计算密码哈希避免bcrypt兼容性问题
"""

import sys
import uvicorn
from datetime import datetime, timedelta
from typing import List, Optional
import hashlib
import hmac

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from jose import JWTError, jwt
import uuid
import time

# 添加src到路径
sys.path.insert(0, 'src')

# 导入配置和日志
from aegis_isle.core.config import settings
from aegis_isle.core.logging import logger, audit_logger

# ==================== 配置 ====================

SECRET_KEY = settings.secret_key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = settings.access_token_expire_minutes

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")

# 简单密码哈希上下文 (HMAC-SHA256)
class SimplePwdContext:
    def hash(self, password: str) -> str:
        return hmac.new(
            b"salt-key-aegis-isle",
            password.encode(),
            hashlib.sha256
        ).hexdigest()

    def verify(self, password: str, hash_value: str) -> bool:
        return self.hash(password) == hash_value

pwd_context = SimplePwdContext()

# ==================== 数据模型 ====================

class User(BaseModel):
    user_id: str
    username: str
    email: str
    full_name: str
    hashed_password: str
    is_active: bool = True
    roles: List[str] = ["user"]

class TokenData(BaseModel):
    username: str
    roles: List[str]

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    user_info: dict

class CurrentUser(BaseModel):
    user_id: str
    username: str
    email: str
    full_name: str
    roles: List[str]
    is_active: bool

# ==================== 用户数据库 ====================

# 预计算的密码哈希
ADMIN_HASH = "2d65adf71e4a3da0890e7a35926595b14e70a3f01d0d720b88c44b12e1e86fba"
TESTUSER_HASH = "c0c02e4069d3f05e79d27548d4ef357e07ce8d02ddd01c3ff8873eb8a34cd454"

USERS_DB = {
    "admin": User(
        user_id="admin_001",
        username="admin",
        email="admin@aegis-isle.dev",
        full_name="Administrator",
        hashed_password=ADMIN_HASH,
        is_active=True,
        roles=["user", "admin", "super_admin"]
    ),
    "testuser": User(
        user_id="user_001",
        username="testuser",
        email="testuser@aegis-isle.dev",
        full_name="Test User",
        hashed_password=TESTUSER_HASH,
        is_active=True,
        roles=["user"]
    )
}

# ==================== 认证功能 ====================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_user(username: str) -> Optional[User]:
    return USERS_DB.get(username)

def authenticate_user(username: str, password: str) -> Optional[User]:
    user = get_user(username)
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> TokenData:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        roles: list = payload.get("roles", [])
        if username is None:
            raise credentials_exception
        return TokenData(username=username, roles=roles)
    except JWTError:
        raise credentials_exception

async def get_current_user(token: str = Depends(oauth2_scheme)) -> CurrentUser:
    token_data = verify_token(token)
    user = get_user(token_data.username)
    if user is None or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")
    return CurrentUser(
        user_id=user.user_id,
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        roles=user.roles,
        is_active=user.is_active
    )

def require_admin(current_user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    if "admin" not in current_user.roles and "super_admin" not in current_user.roles:
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return current_user

def require_super_admin(current_user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    if "super_admin" not in current_user.roles:
        raise HTTPException(status_code=403, detail="Super admin privileges required")
    return current_user

# ==================== FastAPI 应用 ====================

app = FastAPI(
    title="AegisIsle 认证系统演示",
    description="OAuth2 + RBAC + 审计日志演示服务器",
    version="1.0.0"
)

# ==================== 中间件 ====================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """请求日志中间件"""
    request_id = str(uuid.uuid4())
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")

    start_time = time.time()
    logger.info(f"Request {request_id}: {request.method} {request.url.path} from {client_ip}")

    response = await call_next(request)

    duration = time.time() - start_time
    duration_ms = duration * 1000

    logger.info(f"Request {request_id}: {response.status_code} ({duration:.3f}s)")

    # 记录API访问审计日志
    if str(request.url.path).startswith("/api/"):
        audit_logger.log_api_access(
            method=request.method,
            endpoint=request.url.path,
            ip_address=client_ip,
            user_agent=user_agent,
            status_code=response.status_code,
            response_time_ms=duration_ms,
            request_id=request_id
        )

    response.headers["X-Request-ID"] = request_id
    return response

# ==================== API 端点 ====================

@app.get("/")
async def root():
    """根端点"""
    return {
        "message": "AegisIsle 认证系统演示服务器",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "login": "POST /api/v1/auth/token",
            "me": "GET /api/v1/auth/me",
            "status": "GET /api/v1/auth/status",
            "admin_test": "GET /api/v1/auth/admin-test",
            "super_admin_test": "GET /api/v1/auth/super-admin-test"
        },
        "default_users": {
            "admin": "admin/admin123 (super_admin)",
            "testuser": "testuser/testpass123 (user)"
        }
    }

@app.get("/api/v1/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "aegis-isle-auth"
    }

@app.post("/api/v1/auth/token", response_model=TokenResponse)
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends()
):
    """
    OAuth2兼容的登录端点

    - **username**: 用户名 (admin 或 testuser)
    - **password**: 密码 (admin123 或 testpass123)
    """
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")

    # 认证用户
    user = authenticate_user(form_data.username, form_data.password)

    if not user:
        # 记录失败的认证尝试
        audit_logger.log_authentication(
            action="login_failed",
            username=form_data.username,
            outcome="failure",
            ip_address=client_ip,
            user_agent=user_agent,
            error_message="Invalid credentials"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 创建访问令牌
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "roles": user.roles},
        expires_delta=access_token_expires
    )

    # 记录成功的认证
    audit_logger.log_authentication(
        action="login_success",
        username=user.username,
        outcome="success",
        ip_address=client_ip,
        user_agent=user_agent
    )

    logger.info(f"User {user.username} logged in successfully")

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user_info={
            "user_id": user.user_id,
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
            "roles": user.roles
        }
    )

@app.get("/api/v1/auth/me")
async def get_me(current_user: CurrentUser = Depends(get_current_user)):
    """获取当前用户信息"""
    return {
        "user_id": current_user.user_id,
        "username": current_user.username,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "roles": current_user.roles,
        "is_active": current_user.is_active
    }

@app.get("/api/v1/auth/status")
async def get_auth_status(current_user: CurrentUser = Depends(get_current_user)):
    """获取认证状态和权限"""

    # 计算用户权限
    USER_ROLES = {
        "user": ["users:read", "system:read"],
        "admin": ["users:read", "users:write", "system:read", "system:write", "audit:read"],
        "super_admin": ["users:read", "users:write", "system:read", "system:write", "audit:read", "audit:write"],
    }

    permissions = []
    for role in current_user.roles:
        permissions.extend(USER_ROLES.get(role, []))
    permissions = list(set(permissions))  # 去重

    return {
        "authenticated": True,
        "user": {
            "user_id": current_user.user_id,
            "username": current_user.username,
            "email": current_user.email,
            "full_name": current_user.full_name
        },
        "roles": current_user.roles,
        "permissions": sorted(permissions),
        "is_admin": "admin" in current_user.roles or "super_admin" in current_user.roles,
        "is_super_admin": "super_admin" in current_user.roles
    }

@app.get("/api/v1/auth/admin-test")
async def admin_test(admin_user: CurrentUser = Depends(require_admin)):
    """管理员权限测试端点"""
    return {
        "message": "Admin access granted!",
        "user": admin_user.username,
        "roles": admin_user.roles,
        "endpoint": "admin-test"
    }

@app.get("/api/v1/auth/super-admin-test")
async def super_admin_test(super_admin_user: CurrentUser = Depends(require_super_admin)):
    """超级管理员权限测试端点"""
    return {
        "message": "Super admin access granted!",
        "user": super_admin_user.username,
        "roles": super_admin_user.roles,
        "endpoint": "super-admin-test"
    }

@app.post("/api/v1/auth/logout")
async def logout(current_user: CurrentUser = Depends(get_current_user)):
    """登出端点"""
    audit_logger.log_authentication(
        action="logout",
        username=current_user.username,
        outcome="success"
    )

    return {
        "message": "Successfully logged out",
        "username": current_user.username,
        "note": "Token invalidation should be implemented client-side"
    }

# ==================== 启动 ====================

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║  AegisIsle OAuth2 + RBAC + 审计日志演示服务器                 ║
    ╚════════════════════════════════════════════════════════════════╝

    服务器启动中...

    默认用户账号:
      1. admin / admin123 (super_admin权限)
      2. testuser / testpass123 (user权限)

    访问地址:
      - API文档: http://localhost:8000/docs
      - ReDoc: http://localhost:8000/redoc
      - 根端点: http://localhost:8000/
      - 健康检查: http://localhost:8000/api/v1/health

    审计日志位置:
      - logs/audit/audit_YYYY-MM-DD.jsonl

    测试命令:
      # 登录获取token
      curl -X POST "http://localhost:8000/api/v1/auth/token" \\
        -H "Content-Type: application/x-www-form-urlencoded" \\
        -d "username=admin&password=admin123"

      # 使用token访问受保护端点
      curl -X GET "http://localhost:8000/api/v1/auth/me" \\
        -H "Authorization: Bearer YOUR_TOKEN"

    按 Ctrl+C 停止服务器
    ═══════════════════════════════════════════════════════════════════
    """)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
