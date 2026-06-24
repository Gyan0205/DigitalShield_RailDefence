"""
Digital Shield Rail Defense -- Authentication & Authorization
================================================================
API key authentication with role-based access control.

Roles:
  - admin:   Full access to all endpoints
  - officer: Alerts, risk scores, anomaly detection
  - viewer:  Read-only access to metadata and train lookups

Keys are loaded from environment variables (DS_API_KEY_ADMIN,
DS_API_KEY_OFFICER, DS_API_KEY_VIEWER) or fall back to dev defaults.
"""

import os
import logging
from typing import Optional, Dict

from fastapi import HTTPException, Security, Depends
from fastapi.security import APIKeyHeader

logger = logging.getLogger("auth")

# API key header scheme
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# ── Key Store ──────────────────────────────────────────────
# Production: set DS_API_KEY_ADMIN, DS_API_KEY_OFFICER, DS_API_KEY_VIEWER in .env
# Development: uses insecure defaults below (only when ENVIRONMENT != production)

_env = os.getenv("ENVIRONMENT", "development")

def _build_key_store() -> Dict[str, Dict]:
    """Build API key store from environment variables."""
    store = {}

    # Environment-defined keys (preferred)
    env_keys = {
        "DS_API_KEY_ADMIN": {"role": "admin", "name": "Admin"},
        "DS_API_KEY_OFFICER": {"role": "officer", "name": "RPF Officer"},
        "DS_API_KEY_VIEWER": {"role": "viewer", "name": "Dashboard Viewer"},
        "DS_API_KEY": {"role": "admin", "name": "Env Admin"},
    }
    for env_var, meta in env_keys.items():
        key = os.getenv(env_var)
        if key:
            store[key] = {**meta, "active": True, "source": "env"}

    # Development defaults (NEVER use in production)
    if _env != "production":
        dev_defaults = {
            "ds-dev-key": {"role": "admin", "name": "Dev Admin", "active": True, "source": "default"},
            "ds-officer-dev": {"role": "officer", "name": "Dev Officer", "active": True, "source": "default"},
            "ds-viewer-dev": {"role": "viewer", "name": "Dev Viewer", "active": True, "source": "default"},
        }
        for k, v in dev_defaults.items():
            if k not in store:
                store[k] = v

    if store:
        logger.info(f"Auth: {len(store)} API keys loaded ({sum(1 for v in store.values() if v.get('source') == 'env')} from env)")
    else:
        logger.warning("Auth: No API keys configured. Development fallback will be used.")

    return store


API_KEYS = _build_key_store()

# Role hierarchy
ROLE_HIERARCHY = {
    "admin": {"admin", "officer", "viewer"},
    "officer": {"officer", "viewer"},
    "viewer": {"viewer"},
}


async def get_current_user(api_key: Optional[str] = Security(API_KEY_HEADER)) -> Dict:
    """
    Validate API key and return user context.
    In development mode (no key provided), returns a dev user.
    """
    if not api_key:
        if _env != "production":
            return {"role": "admin", "name": "Dev User (no auth)", "authenticated": False}
        raise HTTPException(status_code=401, detail="API key required. Set X-API-Key header.")

    key_info = API_KEYS.get(api_key)
    if not key_info or not key_info.get("active"):
        raise HTTPException(status_code=403, detail="Invalid or inactive API key")

    return {
        "role": key_info["role"],
        "name": key_info["name"],
        "authenticated": True,
    }


def require_role(required_role: str):
    """Dependency that enforces role-based access."""
    async def _check(user: Dict = Depends(get_current_user)) -> Dict:
        user_role = user.get("role", "viewer")
        allowed = ROLE_HIERARCHY.get(user_role, set())
        if required_role not in allowed:
            raise HTTPException(
                status_code=403,
                detail=f"Role '{user_role}' cannot access '{required_role}' endpoints",
            )
        return user
    return _check
