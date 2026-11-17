"""Pydantic schemas for authentication and authorization."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator
import re


EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")


class UserRegisterRequest(BaseModel):
    """Registration request with email and password."""
    email: str = Field(..., description="User email address", examples=["user@example.com"])
    password: str = Field(..., min_length=6, description="User password (min 6 characters)", examples=["SecureP@ss123"])
    full_name: Optional[str] = Field(None, description="Optional full name of the user", examples=["Jane Doe"])

    @field_validator("email")
    @classmethod
    def validate_email_format(cls, v: str) -> str:
        clean = v.strip().lower()
        if not EMAIL_REGEX.match(clean):
            raise ValueError("Invalid email address format.")
        return clean


class UserLoginRequest(BaseModel):
    """Login request with email and password."""
    email: str = Field(..., description="User email address", examples=["user@example.com"])
    password: str = Field(..., description="User password", examples=["SecureP@ss123"])

    @field_validator("email")
    @classmethod
    def normalize_email(cls, v: str) -> str:
        return v.strip().lower()


class UserResponse(BaseModel):
    """Public user response schema."""
    id: str = Field(..., description="User unique MongoDB ID")
    email: str = Field(..., description="User email address")
    full_name: Optional[str] = Field(None, description="User full name")
    role: str = Field("user", description="User role: 'user' or 'admin'")
    is_active: bool = Field(True, description="Account active status")
    created_at: datetime = Field(..., description="Account creation timestamp")


class TokenResponse(BaseModel):
    """Authentication token response with access and refresh tokens."""
    access_token: str = Field(..., description="JWT Bearer access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(..., description="Access token expiration duration in seconds")
    user: UserResponse = Field(..., description="Authenticated user profile")


class RefreshTokenRequest(BaseModel):
    """Token refresh request."""
    refresh_token: str = Field(..., description="JWT refresh token to exchange for new access token")


class RefreshTokenResponse(BaseModel):
    """Response returned upon successful token refresh."""
    access_token: str = Field(..., description="New JWT Bearer access token")
    refresh_token: str = Field(..., description="New (or rotated) JWT refresh token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(..., description="Access token expiration duration in seconds")


class MessageResponse(BaseModel):
    """Generic status response message."""
    message: str = Field(..., description="Operation status description")
    success: bool = Field(True, description="Status flag")
