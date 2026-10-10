"""Facebook Marketplace through the owner's own browser session. See flow_notes for the flows."""

from marketplace_agent.ai.factory import build_router
from marketplace_agent.config import Settings
from marketplace_agent.core.clock import Clock
from marketplace_agent.core.db import make_engine
from marketplace_agent.platforms.base import PlatformAuthError
from marketplace_agent.platforms.facebook.adapter import FacebookAdapter
from marketplace_agent.platforms.facebook.driver import PacedDriver, PlaywrightDriver
from marketplace_agent.platforms.facebook.pacing import Pacer

__all__ = ["FacebookAdapter", "build_adapter"]


def build_adapter(settings: Settings, clock: Clock) -> FacebookAdapter:
    """Factory for core.runtime.build_platforms. The browser starts on the first call that
    needs it; the router is only used to heal selectors. PlatformAuthError when no browser
    profile exists yet, so nothing opens an empty, signed-out browser."""
    if not settings.fb_profile_dir.is_dir():
        raise PlatformAuthError(
            f"Facebook is not signed in on this machine (no browser profile at "
            f"{settings.fb_profile_dir}): run `agent facebook login` once, or remove 'facebook' "
            "from enabled_platforms."
        )
    pacer = Pacer(settings.facebook_action_delay_s)
    browser = PlaywrightDriver(
        settings.fb_profile_dir, headless=settings.facebook_headless, pacer=pacer
    )
    # The runtime has created the tables already; connecting only on a heal keeps
    # `agent doctor` from creating a database.
    router = build_router(settings, make_engine(settings.database_url), clock)
    return FacebookAdapter(settings, clock, PacedDriver(browser, pacer), router)
