"""In-memory marketplace with scripted buyers, for tests and simulation."""
import secrets

from marketplace_agent.config import Settings
from marketplace_agent.core.clock import Clock
from marketplace_agent.platforms.fake.buyers import (
    AvailabilityAsker,
    Bundler,
    FullPrice,
    Ghoster,
    Injector,
    Lowballer,
    Negotiator,
    NoShow,
    OtherListing,
    Persona,
    Questioner,
    ReasonableBuyer,
    Scammer,
    Scripted,
    persona_class,
)
from marketplace_agent.platforms.fake.marketplace import FakeMarketplace, Oracle

__all__ = [
    "AvailabilityAsker",
    "Bundler",
    "FakeMarketplace",
    "FullPrice",
    "Ghoster",
    "Injector",
    "Lowballer",
    "Negotiator",
    "NoShow",
    "Oracle",
    "OtherListing",
    "Persona",
    "Questioner",
    "ReasonableBuyer",
    "Scammer",
    "Scripted",
    "build_adapter",
    "persona_class",
]


def build_adapter(settings: Settings, clock: Clock) -> FakeMarketplace:
    """Factory for core.runtime.build_platforms. The fake platform reads no settings. Each CLI
    command is a new process over the same database, so its post ids get a random prefix."""
    return FakeMarketplace(clock, id_prefix=f"{secrets.token_hex(3)}-")
