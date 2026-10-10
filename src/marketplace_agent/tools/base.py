"""Tool contract. The model chooses tools; guard.validate decides whether they may run."""
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from pydantic import BaseModel
from sqlmodel import Session

from marketplace_agent.config import Settings
from marketplace_agent.core.clock import Clock
from marketplace_agent.notify.base import Notifier
from marketplace_agent.platforms.base import PlatformAdapter


@dataclass(frozen=True)
class Allow:
    pass


@dataclass(frozen=True)
class Reject:
    reason: str  # returned to the model verbatim so it can choose differently


@dataclass(frozen=True)
class NeedsApproval:
    reason: str  # trust tier 0: park as PendingApproval and ask the human


Verdict = Allow | Reject | NeedsApproval


@dataclass
class ToolResult:
    ok: bool
    summary: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolContext:
    session: Session
    clock: Clock
    settings: Settings
    platforms: Mapping[str, PlatformAdapter]
    notifier: Notifier
    item_id: int
    agent_run_id: int | None = None
    actor: str = "agent"  # agent | human | system


class Tool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    Args: ClassVar[type[BaseModel]]
    trust_action: ClassVar[str | None] = None  # trust ladder action key; None = not gated
    read_only: ClassVar[bool] = False

    def parse_args(self, raw: dict[str, Any]) -> BaseModel:
        return self.Args.model_validate(raw)

    def trust_platform(self, ctx: ToolContext, args: BaseModel) -> str:
        """Platform the trust ladder should consult for this call. '*' = platform independent."""
        return "*"

    @abstractmethod
    def check(self, ctx: ToolContext, args: BaseModel) -> Verdict:
        """Tool-specific invariants. Must not mutate state."""

    @abstractmethod
    def execute(self, ctx: ToolContext, args: BaseModel) -> ToolResult:
        """Perform the action. Only called after guard.validate returned Allow."""

    @classmethod
    def spec(cls) -> dict[str, Any]:
        return {
            "name": cls.name,
            "description": cls.description,
            "args_schema": cls.Args.model_json_schema(),
        }
