"""Runtime contract checks for real DAG T27 dataflow."""

import logging
from dataclasses import dataclass

import pytest

from src.orchestration.real_dag_orchestrator import DAGNode, RealDAGOrchestrator


@dataclass
class _FakeToolResult:
    data: dict


class _FakeT27Tool:
    """Capture T27 requests built by the DAG orchestrator."""

    def __init__(self) -> None:
        self.requests = []

    def execute(self, request):
        self.requests.append(request)
        return _FakeToolResult(data={"relationships": [], "total_found": 0})


@pytest.mark.asyncio
async def test_t27_node_receives_entities_from_upstream_ner_node() -> None:
    """T27 DAG nodes should receive both chunk text and entity records."""
    fake_tool = _FakeT27Tool()
    orchestrator = object.__new__(RealDAGOrchestrator)
    orchestrator.nodes = {
        "chunk_text": DAGNode(
            node_id="chunk_text",
            tool_name="T15A_TEXT_CHUNKER",
            tool_instance=object(),
            inputs=[],
            parameters={},
            status="completed",
            result={
                "chunks": [
                    {
                        "chunk_ref": "chunk-1",
                        "text": "Alice works at Acme Corp.",
                    }
                ]
            },
        ),
        "extract_entities": DAGNode(
            node_id="extract_entities",
            tool_name="T23A_SPACY_NER",
            tool_instance=object(),
            inputs=[],
            parameters={},
            status="completed",
            result={
                "entities": [
                    {
                        "surface_form": "Alice",
                        "entity_type": "PERSON",
                        "start_pos": 0,
                        "end_pos": 5,
                    },
                    {
                        "surface_form": "Acme Corp",
                        "entity_type": "ORG",
                        "start_pos": 15,
                        "end_pos": 24,
                    },
                ]
            },
        ),
        "extract_relations": DAGNode(
            node_id="extract_relations",
            tool_name="T27_RELATIONSHIP_EXTRACTOR",
            tool_instance=fake_tool,
            inputs=["chunk_text", "extract_entities"],
            parameters={},
        ),
    }
    orchestrator.provenance = []
    orchestrator.logger = logging.getLogger("test-real-dag")

    result = await orchestrator.execute_node("extract_relations", {})

    assert result == {"relationships": [], "total_found": 0}
    sent_input = fake_tool.requests[0].input_data
    assert sent_input["chunk_ref"] == "chunk-1"
    assert sent_input["text"] == "Alice works at Acme Corp."
    assert sent_input["entities"] == orchestrator.nodes["extract_entities"].result["entities"]
