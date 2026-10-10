"""Settings from env (MA_ prefix), .env, and marketplace_agent.toml, in that priority order."""
from datetime import time
from pathlib import Path

from pydantic import AliasChoices, BaseModel, Field, SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class AvailabilityWindow(BaseModel):
    weekday: int = Field(ge=0, le=6)  # 0 = Monday, local time
    start: time
    end: time


class ProviderLimit(BaseModel):
    rpm: int
    rpd: int


def _default_availability() -> list[AvailabilityWindow]:
    weekday = [AvailabilityWindow(weekday=d, start=time(17), end=time(20)) for d in range(5)]
    weekend = [AvailabilityWindow(weekday=d, start=time(10), end=time(18)) for d in (5, 6)]
    return weekday + weekend


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MA_",
        env_file=".env",
        env_nested_delimiter="__",
        toml_file="marketplace_agent.toml",
        extra="ignore",
    )

    # storage
    data_dir: Path = Path("data")
    db_url: str | None = None

    # seller
    timezone: str = "America/Toronto"
    availability: list[AvailabilityWindow] = Field(default_factory=_default_availability)
    pickup_area: str = ""        # cross street, safe to share any time
    pickup_address: str = ""     # exact address, shared only after a confirmed commitment
    enabled_platforms: list[str] = Field(default_factory=lambda: ["fake"])

    # guardrails
    daily_outbound_cap: dict[str, int] = Field(
        default_factory=lambda: {"fake": 1000, "kijiji": 60, "facebook": 40}
    )
    daily_publish_cap: dict[str, int] = Field(
        default_factory=lambda: {"fake": 1000, "kijiji": 10, "facebook": 5}
    )
    undo_window_minutes: int = 10
    trust_promote_after: int = 5
    approval_ttl_hours: int = 24

    # runner
    tick_seconds: int = 120
    idle_wake_days: float = 3.0
    stats_wake_view_delta: int = 25
    no_show_grace_minutes: int = 30

    # models (free tier by default)
    gemini_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("MA_GEMINI_API_KEY", "GEMINI_API_KEY")
    )
    groq_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("MA_GROQ_API_KEY", "GROQ_API_KEY")
    )
    openrouter_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("MA_OPENROUTER_API_KEY", "OPENROUTER_API_KEY")
    )
    vision_model: str = "gemini-flash-latest"
    reasoning_model: str = "gemini-flash-latest"
    cheap_model: str = "gemini-3.5-flash-lite"  # alias target unpublished; pinned
    groq_model: str = "openai/gpt-oss-120b"
    openrouter_model: str = "nvidia/nemotron-3-super-120b-a12b:free"
    provider_limits: dict[str, ProviderLimit] = Field(
        default_factory=lambda: {
            "gemini": ProviderLimit(rpm=10, rpd=250),
            "groq": ProviderLimit(rpm=30, rpd=1000),
            "openrouter": ProviderLimit(rpm=20, rpd=50),
        }
    )
    quota_constrained_fraction: float = 0.25  # below this share of daily budget left: constrained

    # discord
    discord_token: SecretStr | None = None
    discord_channel_id: int | None = None
    discord_owner_id: int | None = None
    discord_digest_hour: int = Field(default=20, ge=0, le=23)  # local time

    # kijiji
    kijiji_email: str | None = None
    kijiji_password: SecretStr | None = None
    kijiji_location_id: int | None = None
    kijiji_postal_code: str | None = None

    # facebook
    facebook_profile_dir: Path | None = None  # default: data_dir / "facebook_profile"
    facebook_headless: bool = False
    facebook_heal_budget_per_day: int = 5
    facebook_action_delay_s: tuple[float, float] = (1.5, 4.0)  # human pacing between browser actions

    # web intake
    web_host: str = "127.0.0.1"
    web_port: int = 8765
    web_token: SecretStr | None = None  # when set, the web UI requires it
    web_max_photos: int = 12
    web_max_photo_mb: int = 15

    @property
    def database_url(self) -> str:
        return self.db_url or f"sqlite:///{self.data_dir / 'agent.db'}"

    @property
    def photos_dir(self) -> Path:
        return self.data_dir / "photos"

    @property
    def fb_profile_dir(self) -> Path:
        return self.facebook_profile_dir or self.data_dir / "facebook_profile"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )
