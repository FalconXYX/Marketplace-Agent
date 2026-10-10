"""Deterministic provider for tests and simulation. Replays a script instead of calling a model."""
import json
from collections import deque
from collections.abc import Callable, Sequence
from typing import Any

from marketplace_agent.ai.base import Capability, LLMRequest, LLMResponse, ProviderError
from marketplace_agent.ai.router import extract_json

Reply = dict[str, Any] | str
Responder = Callable[[LLMRequest], Reply]
ScriptEntry = Reply | Exception | Responder


class ScriptedProvider:
    """Each request consumes the next script entry, or calls the script if it is a callable.

    | Entry | Result |
    |---|---|
    | dict | `data` is the dict, `text` its JSON |
    | str | `text`; `data` parsed from it when the request has a `json_schema` |
    | Exception | raised |
    | callable | called with the request, result handled as above |
    """

    def __init__(
        self,
        script: Sequence[ScriptEntry] | Responder,
        *,
        name: str = "scripted",
        capabilities: frozenset[Capability] = frozenset(Capability),
        model: str = "scripted-1",
    ):
        self.name = name
        self.capabilities = capabilities
        self.model = model
        self.requests: list[LLMRequest] = []
        self._responder = script if callable(script) else None
        self._queue: deque[ScriptEntry] = deque() if callable(script) else deque(script)

    def model_for(self, capability: Capability) -> str:
        return self.model

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        entry = self._next_entry()
        if callable(entry):
            entry = entry(request)
        if isinstance(entry, Exception):
            raise entry
        return self._respond(request, entry)

    def _next_entry(self) -> ScriptEntry:
        if self._responder is not None:
            return self._responder
        if not self._queue:
            raise ProviderError("script exhausted")
        return self._queue.popleft()

    def _respond(self, request: LLMRequest, reply: Reply) -> LLMResponse:
        model = self.model_for(request.capability)
        if isinstance(reply, dict):
            return LLMResponse(text=json.dumps(reply), provider=self.name, model=model, data=reply)
        data = extract_json(reply) if request.json_schema is not None else None
        return LLMResponse(text=reply, provider=self.name, model=model, data=data)
