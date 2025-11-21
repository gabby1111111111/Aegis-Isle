"""
Dependency injection for FastAPI endpoints with OAuth2 + RBAC authentication.
"""

import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from ..agents.orchestrator import AgentOrchestrator
from ..agents.router import AgentRouter
from ..core.config import settings
from ..core.logging import logger, audit_logger
from ..rag.pipeline import RAGPipeline


# OAuth2 and RBAC Models
class TokenData(BaseModel):
    """Token payload data model."""
    username: Optional[str] = None
    user_id: Optional[str] = None
    roles: List[str] = []
    exp: Optional[datetime] = None


class UserInDB(BaseModel):
    """User model for database storage."""
    username: str
    user_id: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    hashed_password: str
    roles: List[str] = ["user"]
    is_active: bool = True
    created_at: datetime
    last_login: Optional[datetime] = None


class CurrentUser(BaseModel):
    """Current authenticated user model."""
    user_id: str
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    roles: List[str] = ["user"]
    is_active: bool = True


# OAuth2 Configuration
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/token",
    scopes={
        "read": "Read access",
        "write": "Write access",
        "admin": "Administrative access"
    }
)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Mock database for users (in production, use real database)
# Use lazy initialization to avoid bcrypt issues during module import
_USERS_DB: Optional[Dict[str, UserInDB]] = None

def get_users_db() -> Dict[str, UserInDB]:
    """Get user database, initializing if needed."""
    global _USERS_DB
    if _USERS_DB is None:
        _USERS_DB = {
            # Default admin user (configurable via environment variables)
            os.getenv("ADMIN_USERNAME", "admin"): UserInDB(
                username=os.getenv("ADMIN_USERNAME", "admin"),
                user_id="admin_001",
                email="admin@aegisisle.com",
                full_name="System Administrator",
                hashed_password=pwd_context.hash(os.getenv("ADMIN_PASSWORD", "admin123")),
                roles=["user", "admin", "super_admin"],
                is_active=True,
                created_at=datetime.utcnow()
            ),
            # Default test user
            "testuser": UserInDB(
                username="testuser",
                user_id="user_001",
                email="test@aegisisle.com",
                full_name="Test User",
                hashed_password=pwd_context.hash("testpass123"),
                roles=["user"],
                is_active=True,
                created_at=datetime.utcnow()
            )
        }
    return _USERS_DB


# Authentication Functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password for storage."""
    return pwd_context.hash(password)


def get_user(username: str) -> Optional[UserInDB]:
    """Get user from database by username."""
    return get_users_db().get(username)


def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """Authenticate user with username and password."""
    user = get_user(username)
    if not user:
        # Log failed authentication - user not found
        audit_logger.log_authentication(
            action="login_attempt",
            username=username,
            outcome="failure",
            error_message="User not found"
        )
        return None
    if not verify_password(password, user.hashed_password):
        # Log failed authentication - invalid password
        audit_logger.log_authentication(
            action="login_attempt",
            username=username,
            outcome="failure",
            error_message="Invalid password"
        )
        return None
    if not user.is_active:
        # Log failed authentication - inactive user
        audit_logger.log_authentication(
            action="login_attempt",
            username=username,
            outcome="failure",
            error_message="User account is inactive"
        )
        return None

    # Update last login
    user.last_login = datetime.utcnow()

    # Log successful authentication
    audit_logger.log_authentication(
        action="login_success",
        username=username,
        outcome="success"
    )

    return user


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token.

    Args:
        data: Token payload data (typically user info)
        expires_delta: Token expiration time override

    Returns:
        Encoded JWT token string

    Raises:
        HTTPException: If token creation fails
    """
    try:
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)

        to_encode.update({"exp": expire})

        # Add token metadata
        to_encode.update({
            "iat": datetime.utcnow(),
            "iss": "aegis-isle",
            "type": "access_token"
        })

        encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm="HS256")

        logger.info(f"Access token created for user: {data.get('username')}")
        return encoded_jwt

    except Exception as e:
        logger.error(f"Failed to create access token: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not create access token"
        )


def verify_token(token: str) -> TokenData:
    """Verify and decode JWT token.

    Args:
        token: JWT token string

    Returns:
        TokenData with user information

    Raises:
        HTTPException: If token is invalid or expired
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        # Decode JWT token
        payload = jwt.decode(token, settings.secret_key, algorithms=["HS256"])

        # Extract user information
        username: str = payload.get("username")
        user_id: str = payload.get("user_id")
        roles: List[str] = payload.get("roles", [])
        exp: datetime = datetime.fromtimestamp(payload.get("exp", 0))

        if username is None or user_id is None:
            logger.warning("Token missing required fields")
            raise credentials_exception

        # Check token expiration
        if exp < datetime.utcnow():
            logger.warning(f"Expired token for user: {username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )

        token_data = TokenData(
            username=username,
            user_id=user_id,
            roles=roles,
            exp=exp
        )

        logger.debug(f"Token verified for user: {username}")
        return token_data

    except JWTError as e:
        logger.warning(f"JWT decode error: {e}")
        raise credentials_exception
    except Exception as e:
        logger.error(f"Token verification failed: {e}")
        raise credentials_exception


async def get_current_user(token: str = Depends(oauth2_scheme)) -> CurrentUser:
    """Get current authenticated user from JWT token.

    Args:
        token: JWT token from Authorization header

    Returns:
        CurrentUser object with user information

    Raises:
        HTTPException: If authentication fails
    """
    # Verify token
    token_data = verify_token(token)

    # Get user from database
    user = get_user(token_data.username)
    if user is None:
        logger.warning(f"User not found: {token_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        logger.warning(f"Inactive user attempted access: {user.username}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )

    return CurrentUser(
        user_id=user.user_id,
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        roles=user.roles,
        is_active=user.is_active
    )


def require_role(required_roles: List[str]):
    """Create a dependency that requires specific roles.

    Args:
        required_roles: List of roles that are allowed access

    Returns:
        Dependency function for FastAPI
    """
    def role_checker(current_user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if not any(role in current_user.roles for role in required_roles):
            logger.warning(
                f"Access denied for user {current_user.username}. "
                f"Required roles: {required_roles}, User roles: {current_user.roles}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {required_roles}"
            )
        return current_user

    return role_checker


def require_admin(current_user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    """Require admin role for access.

    Args:
        current_user: Current authenticated user

    Returns:
        CurrentUser if admin access is granted

    Raises:
        HTTPException: If user lacks admin privileges
    """
    if "admin" not in current_user.roles:
        # Log failed authorization
        audit_logger.log_authorization(
            action="admin_access_denied",
            user_id=current_user.user_id,
            username=current_user.username,
            resource="admin_endpoints",
            outcome="failure",
            required_permissions=["admin"]
        )

        logger.warning(
            f"Admin access denied for user {current_user.username}. "
            f"User roles: {current_user.roles}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrative privileges required"
        )

    # Log successful authorization
    audit_logger.log_authorization(
        action="admin_access_granted",
        user_id=current_user.user_id,
        username=current_user.username,
        resource="admin_endpoints",
        outcome="success",
        required_permissions=["admin"]
    )

    logger.info(f"Admin access granted to user: {current_user.username}")
    return current_user


def require_super_admin(current_user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    """Require super admin role for access.

    Args:
        current_user: Current authenticated user

    Returns:
        CurrentUser if super admin access is granted

    Raises:
        HTTPException: If user lacks super admin privileges
    """
    if "super_admin" not in current_user.roles:
        # Log failed authorization
        audit_logger.log_authorization(
            action="super_admin_access_denied",
            user_id=current_user.user_id,
            username=current_user.username,
            resource="super_admin_endpoints",
            outcome="failure",
            required_permissions=["super_admin"]
        )

        logger.warning(
            f"Super admin access denied for user {current_user.username}. "
            f"User roles: {current_user.roles}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Super administrative privileges required"
        )

    # Log successful authorization
    audit_logger.log_authorization(
        action="super_admin_access_granted",
        user_id=current_user.user_id,
        username=current_user.username,
        resource="super_admin_endpoints",
        outcome="success",
        required_permissions=["super_admin"]
    )

    logger.info(f"Super admin access granted to user: {current_user.username}")
    return current_user


# Application State Dependencies


def get_rag_pipeline(request: Request) -> RAGPipeline:
    """Get the RAG pipeline from the application state."""
    pipeline = getattr(request.app.state, "rag_pipeline", None)
    if pipeline is None:
        logger.error("RAG pipeline not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline not available"
        )
    return pipeline


def get_agent_router(request: Request) -> AgentRouter:
    """Get the agent router from the application state."""
    router = getattr(request.app.state, "agent_router", None)
    if router is None:
        logger.error("Agent router not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent router not available"
        )
    return router


def get_agent_orchestrator(request: Request) -> AgentOrchestrator:
    """Get the agent orchestrator from the application state."""
    orchestrator = getattr(request.app.state, "agent_orchestrator", None)
    if orchestrator is None:
        logger.error("Agent orchestrator not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent orchestrator not available"
        )
    return orchestrator


def get_metrics_middleware(request: Request):
    """Get the metrics middleware from the application state."""
    middleware = getattr(request.app.state, "metrics_middleware", None)
    if middleware is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Metrics not available"
        )
    return middleware


def get_request_id(request: Request) -> str:
    """Get the request ID from the request state."""
    return getattr(request.state, "request_id", "unknown")