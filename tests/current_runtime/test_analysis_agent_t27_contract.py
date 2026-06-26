"""Runtime contract checks for AnalysisAgent relationship extraction inputs."""

import logging
from dataclasses import dataclass

import pytest

from src.orchestration.agents.analysis_agent import AnalysisAgent, _normalize_entities_for_t27


@dataclass
class _FakeMCPResult:
    success: bool
    data: dict
    error: str | None = None


class _FakeMCPAdapter:
    """Capture MCP calls without invoking external services."""

    def __init__(self) -> None:
        self.calls = []

    async def call_tool(self, name: str, payload: dict) -> _FakeMCPResult:
        self.calls.append((name, payload))
        return _FakeMCPResult(
            success=True,
            data={
                "relationships": [
                    {
                        "relationship_type": "WORKS_FOR",
                        "subject": payload["entities"][0],
                        "object": payload["entities"][1],
                    }
                ]
            },
        )


def test_t23a_entities_convert_to_t27_contract() -> None:
    """T23A output should be accepted by the analysis-agent relationship bridge."""
    entities = [
        {
            "surface_form": "Alice",
            "entity_type": "PERSON",
            "start_pos": 0,
            "end_pos": 5,
            "confidence": 0.91,
            "chunk_ref": "chunk-1",
        },
        {
            "surface_form": "Acme Corp",
            "entity_type": "ORG",
            "start_pos": 15,
            "end_pos": 24,
            "chunk_ref": "chunk-1",
        },
    ]

    normalized = _normalize_entities_for_t27(entities)

    assert normalized[0]["text"] == "Alice"
    assert normalized[0]["entity_type"] == "PERSON"
    assert normalized[0]["start"] == 0
    assert normalized[0]["end"] == 5
    assert normalized[0]["confidence"] == 0.91
    assert normalized[0]["surface_form"] == "Alice"
    assert normalized[1]["confidence"] == 0.8


def test_existing_t27_entities_pass_through_with_default_confidence() -> None:
    """Already-normalized T27 entities should remain usable."""
    entities = [
        {"text": "Alice", "entity_type": "PERSON", "start": 0, "end": 5},
        {"text": "Acme Corp", "entity_type": "ORG", "start": 15, "end": 24, "confidence": 0.95},
    ]

    normalized = _normalize_entities_for_t27(entities)

    assert normalized[0] == {
        "text": "Alice",
        "entity_type": "PERSON",
        "start": 0,
        "end": 5,
        "confidence": 0.8,
    }
    assert normalized[1]["confidence"] == 0.95


def test_invalid_entities_fail_loudly() -> None:
    """Unknown entity shapes should not be silently passed to T27."""
    with pytest.raises(ValueError, match="not T27-compatible"):
        _normalize_entities_for_t27([{"name": "Alice", "type": "PERSON"}])


@pytest.mark.asyncio
async def test_relationship_extraction_bridge_sends_t27_entities_and_returns_relationships() -> None:
    """The analysis-agent bridge should convert entities before the MCP call."""
    fake_mcp = _FakeMCPAdapter()
    agent = object.__new__(AnalysisAgent)
    agent.mcp = fake_mcp
    agent.logger = logging.getLogger("test-analysis-agent")
    t23a_entities = [
        {
            "surface_form": "Alice",
            "entity_type": "PERSON",
            "start_pos": 0,
            "end_pos": 5,
            "confidence": 0.91,
        },
        {
            "surface_form": "Acme Corp",
            "entity_type": "ORG",
            "start_pos": 15,
            "end_pos": 24,
            "confidence": 0.93,
        },
    ]

    result = await agent._extract_relationships_from_chunk("chunk-1", "Alice works at Acme Corp.", t23a_entities)

    assert result == {
        "success": True,
        "relationships": [
            {
                "relationship_type": "WORKS_FOR",
                "subject": {
                    "surface_form": "Alice",
                    "entity_type": "PERSON",
                    "start_pos": 0,
                    "end_pos": 5,
                    "confidence": 0.91,
                    "text": "Alice",
                    "start": 0,
                    "end": 5,
                },
                "object": {
                    "surface_form": "Acme Corp",
                    "entity_type": "ORG",
                    "start_pos": 15,
                    "end_pos": 24,
                    "confidence": 0.93,
                    "text": "Acme Corp",
                    "start": 15,
                    "end": 24,
                },
            }
        ],
    }
    assert fake_mcp.calls[0][0] == "extract_relationships"
    sent_entities = fake_mcp.calls[0][1]["entities"]
    assert sent_entities[0]["text"] == "Alice"
    assert sent_entities[0]["start"] == 0
    assert sent_entities[1]["text"] == "Acme Corp"
    assert sent_entities[1]["end"] == 24
