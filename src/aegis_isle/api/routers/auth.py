"""
Authentication and authorization endpoints for OAuth2 + RBAC.
"""

from datetime import timedelta
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from ..dependencies import (
    authenticate_user,
    create_access_token,
    get_current_user,
    require_admin,
    require_super_admin,
    CurrentUser
)
from ...core.config import settings
from ...core.logging import logger


# Response Models
class Token(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str
    expires_in: int
    user_info: Dict[str, Any]


class UserInfo(BaseModel):
    """User information response model."""
    user_id: str
    username: str
    email: str | None = None
    full_name: str | None = None
    roles: list[str]
    is_active: bool


class AuthStatus(BaseModel):
    """Authentication status response model."""
    authenticated: bool
    user: UserInfo | None = None
    permissions: list[str] = []


# Create router
router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends()
) -> Token:
    """
    OAuth2 compatible token login endpoint.

    Authenticate user with username and password, return JWT access token.

    Args:
        form_data: OAuth2 form containing username and password

    Returns:
        JWT access token and user information

    Raises:
        HTTPException: If authentication fails
    """
    # Authenticate user
    user = authenticate_user(form_data.username, form_data.password)

    if not user:
        logger.warning(f"Failed login attempt for username: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create access token
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    token_data = {
        "username": user.username,
        "user_id": user.user_id,
        "roles": user.roles,
        "email": user.email,
    }

    access_token = create_access_token(
        data=token_data,
        expires_delta=access_token_expires
    )

    logger.info(f"Successful login for user: {user.username}")

    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.access_token_expire_minutes * 60,  # Convert to seconds
        user_info={
            "user_id": user.user_id,
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
            "roles": user.roles,
            "is_active": user.is_active
        }
    )


@router.get("/me", response_model=UserInfo)
async def read_users_me(current_user: CurrentUser = Depends(get_current_user)) -> UserInfo:
    """
    Get current authenticated user information.

    Args:
        current_user: Current authenticated user from JWT token

    Returns:
        Current user information
    """
    logger.debug(f"User info requested for: {current_user.username}")

    return UserInfo(
        user_id=current_user.user_id,
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name,
        roles=current_user.roles,
        is_active=current_user.is_active
    )


@router.get("/status", response_model=AuthStatus)
async def get_auth_status(current_user: CurrentUser = Depends(get_current_user)) -> AuthStatus:
    """
    Get detailed authentication status including permissions.

    Args:
        current_user: Current authenticated user from JWT token

    Returns:
        Authentication status with user info and permissions
    """
    # Determine permissions based on roles
    permissions = []
    if "user" in current_user.roles:
        permissions.extend(["read:documents", "create:queries", "read:agents"])
    if "admin" in current_user.roles:
        permissions.extend([
            "write:documents", "delete:documents", "manage:agents",
            "read:logs", "manage:users"
        ])
    if "super_admin" in current_user.roles:
        permissions.extend([
            "manage:system", "read:metrics", "manage:config",
            "delete:users", "manage:permissions"
        ])

    logger.debug(f"Auth status requested for: {current_user.username}")

    return AuthStatus(
        authenticated=True,
        user=UserInfo(
            user_id=current_user.user_id,
            username=current_user.username,
            email=current_user.email,
            full_name=current_user.full_name,
            roles=current_user.roles,
            is_active=current_user.is_active
        ),
        permissions=list(set(permissions))  # Remove duplicates
    )


@router.post("/refresh")
async def refresh_token(current_user: CurrentUser = Depends(get_current_user)) -> Token:
    """
    Refresh access token for authenticated user.

    Args:
        current_user: Current authenticated user from JWT token

    Returns:
        New JWT access token
    """
    # Create new access token with same data
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    token_data = {
        "username": current_user.username,
        "user_id": current_user.user_id,
        "roles": current_user.roles,
        "email": current_user.email,
    }

    access_token = create_access_token(
        data=token_data,
        expires_delta=access_token_expires
    )

    logger.info(f"Token refreshed for user: {current_user.username}")

    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.access_token_expire_minutes * 60,
        user_info={
            "user_id": current_user.user_id,
            "username": current_user.username,
            "email": current_user.email,
            "full_name": current_user.full_name,
            "roles": current_user.roles,
            "is_active": current_user.is_active
        }
    )


@router.get("/admin-test")
async def test_admin_access(admin_user: CurrentUser = Depends(require_admin)) -> Dict[str, str]:
    """
    Test endpoint for admin role verification.

    Args:
        admin_user: Current user verified to have admin role

    Returns:
        Success message for admin access
    """
    logger.info(f"Admin access test successful for user: {admin_user.username}")
    return {
        "message": "Admin access granted successfully",
        "user": admin_user.username,
        "roles": admin_user.roles
    }


@router.get("/super-admin-test")
async def test_super_admin_access(
    super_admin_user: CurrentUser = Depends(require_super_admin)
) -> Dict[str, str]:
    """
    Test endpoint for super admin role verification.

    Args:
        super_admin_user: Current user verified to have super admin role

    Returns:
        Success message for super admin access
    """
    logger.info(f"Super admin access test successful for user: {super_admin_user.username}")
    return {
        "message": "Super admin access granted successfully",
        "user": super_admin_user.username,
        "roles": super_admin_user.roles
    }


@router.post("/logout")
async def logout(current_user: CurrentUser = Depends(get_current_user)) -> Dict[str, str]:
    """
    Logout current user (invalidate token on client side).

    Note: JWT tokens are stateless, so actual invalidation needs to be
    handled on the client side by removing the token.

    Args:
        current_user: Current authenticated user

    Returns:
        Logout confirmation message
    """
    logger.info(f"User logged out: {current_user.username}")

    return {
        "message": "Logout successful",
        "username": current_user.username,
        "note": "Please remove the token from client storage"
    }