"""What one reasoning call returns. Tool args travel as a JSON string so every provider's
structured-output mode can express the schema."""

import json
from typing import Any

from pydantic import BaseModel, Field

INTENTS = ("availability", "question", "offer", "scheduling", "lowball", "scam", "spam", "other")


class PlannedAction(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    why: str = ""
    parse_error: str | None = None  # args_json was not a JSON object


class MessageLabel(BaseModel):
    message_id: int
    intent: str


class Plan(BaseModel):
    rationale: str
    actions: list[PlannedAction] = Field(default_factory=list)
    message_labels: list[MessageLabel] = Field(default_factory=list)

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> "Plan":
        actions = []
        for raw in data.get("actions") or []:
            action = PlannedAction(tool=str(raw.get("tool", "")), why=str(raw.get("why", "")))
            args = raw.get("args_json", raw.get("args", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError as exc:
                    action.parse_error = f"args_json is not valid JSON: {exc}"
                    args = {}
            if not isinstance(args, dict):
                action.parse_error = "args_json must encode a JSON object"
                args = {}
            action.args = args
            actions.append(action)
        labels = [
            MessageLabel(message_id=int(m["message_id"]), intent=str(m["intent"]))
            for m in data.get("message_labels") or []
            if "message_id" in m and "intent" in m
        ]
        return cls(rationale=str(data.get("rationale", "")), actions=actions, message_labels=labels)


PLAN_WIRE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rationale": {"type": "string"},
        "message_labels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "message_id": {"type": "integer"},
                    "intent": {"type": "string", "enum": list(INTENTS)},
                },
                "required": ["message_id", "intent"],
            },
        },
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tool": {"type": "string"},
                    "args_json": {"type": "string"},
                    "why": {"type": "string"},
                },
                "required": ["tool", "args_json"],
            },
        },
    },
    "required": ["rationale", "actions"],
}
