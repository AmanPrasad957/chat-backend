"""Authentication and authorization endpoints."""

import logging
from fastapi import APIRouter, Depends, status

from app.schemas.auth_schema import (
    UserRegisterRequest,
    UserLoginRequest,
    UserResponse,
    TokenResponse,
    RefreshTokenRequest,
    RefreshTokenResponse,
    MessageResponse,
)
from app.services.auth_service import (
    register_user,
    authenticate_user,
    refresh_tokens,
    revoke_refresh_token,
    get_current_user,
)

logger = logging.getLogger("ai_chat_api.routes.auth")

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
    description="Create a new user account with email, password, and optional full name in MongoDB.",
)
async def register(req: UserRegisterRequest) -> UserResponse:
    """Register user with email and hashed password."""
    return await register_user(req)


@router.post(
    "/login",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Authenticate user and obtain JWT tokens",
    description="Verify email & password, and receive a short-lived access token and long-lived refresh token.",
)
async def login(req: UserLoginRequest) -> TokenResponse:
    """Login with email & password."""
    return await authenticate_user(req)


@router.post(
    "/refresh",
    response_model=RefreshTokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Refresh access token",
    description="Exchange a valid refresh token for a fresh access token and rotated refresh token.",
)
async def refresh_token_endpoint(req: RefreshTokenRequest) -> RefreshTokenResponse:
    """Rotate and refresh access token."""
    return await refresh_tokens(req.refresh_token)


@router.post(
    "/logout",
    response_model=MessageResponse,
    status_code=status.HTTP_200_OK,
    summary="Revoke refresh token (Logout)",
    description="Revoke the provided refresh token in MongoDB so it cannot be used again.",
)
async def logout(req: RefreshTokenRequest) -> MessageResponse:
    """Log out user by revoking their refresh token."""
    revoked = await revoke_refresh_token(req.refresh_token)
    return MessageResponse(
        message="Logged out successfully. Token revoked." if revoked else "Token already inactive.",
        success=True,
    )


@router.get(
    "/me",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
    summary="Get current authenticated user profile",
    description="Protected endpoint returning the profile of the user identified by the JWT Bearer access token.",
)
async def get_me(current_user: UserResponse = Depends(get_current_user)) -> UserResponse:
    """Retrieve authenticated user details."""
    return current_user
