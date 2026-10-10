"""Human-facing notifications. Implementations: console, recording (tests), Discord."""
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Protocol


class NoticeKind(StrEnum):
    COMMITMENT = "commitment"  # autonomous deal made; carries an Undo until expires_at
    ESCALATION = "escalation"  # agent deferred to the human
    APPROVAL = "approval"      # trust tier 0 action waiting for approve/reject
    INFO = "info"              # tier 1 act-and-notify
    ALERT = "alert"            # guard trip, cap hit, platform failure
    SOLD = "sold"
    DIGEST = "digest"


@dataclass(frozen=True)
class Notice:
    kind: NoticeKind
    title: str
    body: str
    item_id: int | None = None
    ref_id: int | None = None          # Commitment / Escalation / PendingApproval id
    expires_at: datetime | None = None  # undo deadline for COMMITMENT


class Notifier(Protocol):
    def notify(self, notice: Notice) -> None: ...


class ConsoleNotifier:
    def notify(self, notice: Notice) -> None:
        print(f"[{notice.kind}] {notice.title}: {notice.body}")


@dataclass
class RecordingNotifier:
    notices: list[Notice] = field(default_factory=list)

    def notify(self, notice: Notice) -> None:
        self.notices.append(notice)

    def of_kind(self, kind: NoticeKind) -> list[Notice]:
        return [n for n in self.notices if n.kind == kind]
