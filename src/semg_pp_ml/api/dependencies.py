"""
API dependencies for database, authentication, and services.
"""
from __future__ import annotations

from typing import Generator, Optional
from fastapi import Depends, HTTPException, status, WebSocket, Query
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from jose import jwt, JWTError
from datetime import datetime, timedelta

from ..config import ExperimentConfig
from ..utils.system import get_system_info
from .schemas.auth import TokenData
from .models.db import User, UserRole, get_db as get_db_session


# JWT Configuration
SECRET_KEY = "your-secret-key-change-in-production"  # Should be from env
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


def get_db() -> Generator[Session, None, None]:
    """Database session dependency."""
    db = get_db_session()
    try:
        yield db
    finally:
        db.close()


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.email == token_data.email).first()
    if user is None:
        raise credentials_exception
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return user


async def get_current_user_ws(
    websocket: WebSocket,
    token: str = Query(...),
    db: Session = Depends(get_db)
) -> User:
    """Get current user from WebSocket token query param."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            await websocket.close(code=4001)
            return None
    except JWTError:
        await websocket.close(code=4001)
        return None

    user = db.query(User).filter(User.email == email).first()
    if user is None or not user.is_active:
        await websocket.close(code=4001)
        return None
    return user


def require_role(*roles: UserRole):
    """Dependency to require specific user roles."""
    def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return current_user
    return role_checker


def get_config() -> ExperimentConfig:
    """Get application config from app state."""
    from .main import app
    return app.state.config


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(data: dict) -> str:
    """Create JWT refresh token (longer expiry)."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=7)
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)