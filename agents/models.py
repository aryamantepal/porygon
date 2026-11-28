"""Model clients. The model id is run config; the player never imports a server.

Every call returns a `ModelCall` holding the request and the raw output
verbatim, so a run log can show exactly what the model saw and said.
"""

from __future__ import annotations

import base64
import json
import time
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Message:
    role: str  # "system", "user", or "assistant"
    content: str
    images: Sequence[bytes] = field(default_factory=tuple)  # PNG bytes


@dataclass(frozen=True)
class ModelCall:
    model: str
    messages: Sequence[dict[str, Any]]  # images replaced by a count; the frame is logged separately
    output: str
    thinking: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_seconds: float
    error: str | None = None
    context_length: int | None = None  # the window the server was asked for; a prompt near it may have been cut


class ChatModel(Protocol):
    model_id: str

    def chat(self, messages: Sequence[Message], *, think: bool, stop: Sequence[str] = ()) -> ModelCall: ...


def loggable(messages: Sequence[Message]) -> list[dict[str, Any]]:
    return [
        {"role": m.role, "content": m.content, **({"images": len(m.images)} if m.images else {})}
        for m in messages
    ]


@dataclass
class OllamaChat:
    """A model served by Ollama's /api/chat."""

    model_id: str
    host: str = "http://localhost:11434"
    temperature: float = 0.0
    max_tokens: int = 512
    seed: int | None = 0  # first call's seed; each call adds one, so trials differ but the run replays
    timeout_seconds: float = 600.0
    context_length: int = 16384  # Ollama's default is 4096 and it truncates silently
    _calls: int = field(default=0, init=False, repr=False)

    def chat(self, messages: Sequence[Message], *, think: bool, stop: Sequence[str] = ()) -> ModelCall:
        seed = None if self.seed is None else self.seed + self._calls
        self._calls += 1
        body = {
            "model": self.model_id,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    **({"images": [base64.b64encode(i).decode() for i in m.images]} if m.images else {}),
                }
                for m in messages
            ],
            "stream": False,
            "think": think,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "num_ctx": self.context_length,
                **({"seed": seed} if seed is not None else {}),
                **({"stop": list(stop)} if stop else {}),
            },
        }
        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                reply = json.load(response)
        except Exception as error:  # logged, not raised: a failed call is a data point
            return ModelCall(
                self.model_id, loggable(messages), "", None, None, None,
                time.perf_counter() - started, error=f"{type(error).__name__}: {error}",
            )
        message = reply.get("message", {})
        return ModelCall(
            model=self.model_id,
            messages=loggable(messages),
            output=message.get("content", ""),
            thinking=message.get("thinking") or None,
            prompt_tokens=reply.get("prompt_eval_count"),
            completion_tokens=reply.get("eval_count"),
            latency_seconds=time.perf_counter() - started,
            context_length=self.context_length,
        )


@dataclass
class ScriptedModel:
    """A stand-in model for tests: replies come from a function of the messages."""

    reply: Callable[[Sequence[Message]], str]
    model_id: str = "scripted"
    calls: list[ModelCall] = field(default_factory=list)

    def chat(self, messages: Sequence[Message], *, think: bool, stop: Sequence[str] = ()) -> ModelCall:
        call = ModelCall(self.model_id, loggable(messages), self.reply(messages), None, 0, 0, 0.0)
        self.calls.append(call)
        return call
