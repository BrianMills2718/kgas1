#!/usr/bin/env python3
"""Adapter from KGAS mode selection to Brian's governed llm_client boundary."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class GovernedLLMClientAdapter:
    """Expose the local `complete(prompt)` protocol using `llm_client.acall_llm`.

    ModeSelectionService predates the shared `llm_client` interface and expects a
    small async completion object. This adapter preserves that internal protocol
    while routing actual LLM calls through the governed client with task,
    trace_id, and max_budget metadata.
    """

    model: str = "gpt-5.4-mini"
    task: str = "kgas.mode_selection"
    max_budget: float = 5.0
    last_call_metadata: Dict[str, Any] = field(default_factory=dict)

    async def complete(self, prompt: str) -> str:
        """Return text completion for a mode-selection prompt."""
        from llm_client import acall_llm

        trace_id = self._trace_id(prompt)
        result = await acall_llm(
            self.model,
            [{"role": "user", "content": prompt}],
            task=self.task,
            trace_id=trace_id,
            max_budget=self.max_budget,
        )
        self.last_call_metadata = {
            "model": getattr(result, "model", self.model),
            "requested_model": getattr(result, "requested_model", self.model),
            "resolved_model": getattr(result, "resolved_model", None),
            "trace_id": trace_id,
            "task": self.task,
            "cost": getattr(result, "cost", None),
            "usage": getattr(result, "usage", None),
        }
        return result.content

    def _trace_id(self, prompt: str) -> str:
        """Create a stable, non-secret trace id for observability."""
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        return f"{self.task}:{digest}"

