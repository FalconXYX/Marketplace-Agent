"""Kijiji adapter over the website's own endpoints. api_notes records what is verified."""
from marketplace_agent.config import Settings
from marketplace_agent.core.clock import Clock
from marketplace_agent.platforms.base import PlatformAuthError
from marketplace_agent.platforms.kijiji.adapter import KijijiAdapter
from marketplace_agent.platforms.kijiji.client import KijijiClient

__all__ = ["KijijiAdapter", "KijijiClient", "build_adapter", "build_client"]


def build_client(settings: Settings, clock: Clock) -> KijijiClient:
    """No network until the first call. PlatformAuthError when the account is not configured."""
    email = (settings.kijiji_email or "").strip()
    password = settings.kijiji_password.get_secret_value() if settings.kijiji_password else ""
    if not email or not password:
        raise PlatformAuthError(
            "Kijiji needs an account: set MA_KIJIJI_EMAIL and MA_KIJIJI_PASSWORD (environment, "
            ".env, or marketplace_agent.toml), or remove 'kijiji' from enabled_platforms."
        )
    return KijijiClient(email, password, clock=clock)


def build_adapter(settings: Settings, clock: Clock) -> KijijiAdapter:
    """Factory for core.runtime.build_platforms."""
    return KijijiAdapter(
        build_client(settings, clock),
        clock,
        location_id=settings.kijiji_location_id,
        postal_code=settings.kijiji_postal_code,
    )
