"""Authentication and authorization service with MongoDB, bcrypt, and JWT."""

import os
import uuid
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
from bson import ObjectId
import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config.mongodb import get_database
from app.schemas.auth_schema import (
    UserRegisterRequest,
    UserLoginRequest,
    UserResponse,
    TokenResponse,
    RefreshTokenResponse,
)

logger = logging.getLogger("ai_chat_api.auth")

security_bearer = HTTPBearer(auto_error=False)


def get_jwt_secret() -> str:
    """Retrieve JWT secret key from environment."""
    return os.getenv("JWT_SECRET_KEY", "chat_super_secret_jwt_key_2026_xyz987654321_secure")


def get_jwt_algorithm() -> str:
    """Retrieve JWT signing algorithm."""
    return os.getenv("JWT_ALGORITHM", "HS256")


# ================= PASSWORD SECURITY =================

def hash_password(plain_password: str) -> str:
    """Hash plaintext password using bcrypt with automatic salt."""
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(plain_password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify plaintext password against bcrypt hash."""
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception as exc:
        logger.warning("Password verification failed with exception: %s", exc)
        return False


# ================= JWT TOKEN MANAGEMENT =================

def create_access_token(user_id: str, email: str, role: str = "user") -> Tuple[str, int]:
    """Generate a signed JWT access token. Returns (token, expires_in_seconds)."""
    expire_minutes = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    expires_in_seconds = expire_minutes * 60
    now = datetime.now(timezone.utc)
    expire_at = now + timedelta(minutes=expire_minutes)

    payload = {
        "sub": str(user_id),
        "email": email,
        "role": role,
        "type": "access",
        "iat": int(now.timestamp()),
        "exp": int(expire_at.timestamp()),
        "jti": str(uuid.uuid4()),
    }

    token = jwt.encode(payload, get_jwt_secret(), algorithm=get_jwt_algorithm())
    return token, expires_in_seconds


def create_refresh_token(user_id: str, email: str) -> Tuple[str, datetime]:
    """Generate a signed JWT refresh token. Returns (token, expiration_datetime)."""
    expire_days = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
    now = datetime.now(timezone.utc)
    expire_at = now + timedelta(days=expire_days)

    payload = {
        "sub": str(user_id),
        "email": email,
        "type": "refresh",
        "iat": int(now.timestamp()),
        "exp": int(expire_at.timestamp()),
        "jti": str(uuid.uuid4()),
    }

    token = jwt.encode(payload, get_jwt_secret(), algorithm=get_jwt_algorithm())
    return token, expire_at


def decode_token(token: str, expected_type: str = "access") -> dict:
    """Decode and validate a JWT token's signature, expiration, and type."""
    try:
        payload = jwt.decode(token, get_jwt_secret(), algorithms=[get_jwt_algorithm()])
        token_type = payload.get("type")
        if token_type != expected_type:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token type. Expected '{expected_type}', got '{token_type}'.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"{expected_type.capitalize()} token has expired.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as err:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(err)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ================= AUTHENTICATION ACTIONS =================

async def register_user(req: UserRegisterRequest) -> UserResponse:
    """Register a new user in MongoDB with unique email validation."""
    db = get_database()
    existing = await db.users.find_one({"email": req.email})
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email address is already registered.",
        )

    now = datetime.now(timezone.utc)
    user_doc = {
        "email": req.email,
        "hashed_password": hash_password(req.password),
        "full_name": req.full_name,
        "role": "user",
        "is_active": True,
        "created_at": now,
        "updated_at": now,
    }

    result = await db.users.insert_one(user_doc)
    logger.info("Registered new user '%s' (ID: %s)", req.email, result.inserted_id)

    return UserResponse(
        id=str(result.inserted_id),
        email=user_doc["email"],
        full_name=user_doc["full_name"],
        role=user_doc["role"],
        is_active=user_doc["is_active"],
        created_at=user_doc["created_at"],
    )


async def authenticate_user(req: UserLoginRequest) -> TokenResponse:
    """Authenticate user credentials and issue access + refresh tokens."""
    db = get_database()
    user = await db.users.find_one({"email": req.email})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not verify_password(req.password, user.get("hashed_password", "")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive. Please contact support.",
        )

    user_id = str(user["_id"])
    access_token, expires_in = create_access_token(user_id, user["email"], user.get("role", "user"))
    refresh_token, refresh_exp = create_refresh_token(user_id, user["email"])

    # Store refresh token in MongoDB for revocation and rotation tracking
    await db.refresh_tokens.insert_one({
        "token": refresh_token,
        "user_id": user_id,
        "email": user["email"],
        "expires_at": refresh_exp,
        "created_at": datetime.now(timezone.utc),
        "revoked": False,
    })

    user_resp = UserResponse(
        id=user_id,
        email=user["email"],
        full_name=user.get("full_name"),
        role=user.get("role", "user"),
        is_active=user.get("is_active", True),
        created_at=user.get("created_at", datetime.now(timezone.utc)),
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=expires_in,
        user=user_resp,
    )


async def refresh_tokens(refresh_token_str: str) -> RefreshTokenResponse:
    """Validate refresh token, check revocation, and issue new rotated tokens."""
    payload = decode_token(refresh_token_str, expected_type="refresh")
    user_id = payload.get("sub")
    email = payload.get("email")

    db = get_database()
    stored = await db.refresh_tokens.find_one({"token": refresh_token_str, "revoked": False})
    if not stored:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token is invalid, expired, or has been revoked.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Revoke current token (rotation)
    await db.refresh_tokens.update_one({"token": refresh_token_str}, {"$set": {"revoked": True}})

    # Fetch user to verify account is still active and get current role
    try:
        user_obj_id = ObjectId(user_id)
    except Exception:
        user_obj_id = user_id

    user = await db.users.find_one({"_id": user_obj_id})
    if not user or not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account no longer active.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    new_access_token, expires_in = create_access_token(user_id, email, user.get("role", "user"))
    new_refresh_token, new_refresh_exp = create_refresh_token(user_id, email)

    await db.refresh_tokens.insert_one({
        "token": new_refresh_token,
        "user_id": user_id,
        "email": email,
        "expires_at": new_refresh_exp,
        "created_at": datetime.now(timezone.utc),
        "revoked": False,
    })

    return RefreshTokenResponse(
        access_token=new_access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        expires_in=expires_in,
    )


async def revoke_refresh_token(refresh_token_str: str) -> bool:
    """Revoke a refresh token in MongoDB (logout)."""
    db = get_database()
    res = await db.refresh_tokens.update_one(
        {"token": refresh_token_str},
        {"$set": {"revoked": True, "revoked_at": datetime.now(timezone.utc)}},
    )
    return res.modified_count > 0


# ================= DEPENDENCY: GET_CURRENT_USER =================

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_bearer),
) -> UserResponse:
    """FastAPI dependency to extract, decode, and authenticate current user via Bearer token."""
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_token(credentials.credentials, expected_type="access")
    user_id = payload.get("sub")

    db = get_database()
    try:
        query_id = ObjectId(user_id)
    except Exception:
        query_id = user_id

    user = await db.users.find_one({"_id": query_id})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User associated with token not found.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive.",
        )

    return UserResponse(
        id=str(user["_id"]),
        email=user["email"],
        full_name=user.get("full_name"),
        role=user.get("role", "user"),
        is_active=user.get("is_active", True),
        created_at=user.get("created_at", datetime.now(timezone.utc)),
    )
