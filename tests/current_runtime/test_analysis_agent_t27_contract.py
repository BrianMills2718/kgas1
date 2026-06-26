"""Runtime contract checks for AnalysisAgent relationship extraction inputs."""

import logging
from dataclasses import dataclass

import pytest

from src.orchestration.agents.analysis_agent import AnalysisAgent
from src.analytics.complete_pipeline import CompleteGraphRAGPipeline
from src.tools.compatibility.t27_adapter import normalize_entities_for_t27


@dataclass
class _FakeMCPResult:
    success: bool
    data: dict
    error: str | None = None


@dataclass
class _FakeToolResult:
    status: str
    data: dict
    error_message: str | None = None


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


class _FakeRelationshipExtractor:
    """Capture direct T27 requests from pipeline code."""

    def __init__(self) -> None:
        self.requests = []

    def execute(self, request):
        self.requests.append(request)
        return _FakeToolResult(
            status="success",
            data={
                "relationships": [
                    {
                        "relationship_type": "WORKS_FOR",
                        "subject": request.input_data["entities"][0],
                        "object": request.input_data["entities"][1],
                    }
                ]
            },
        )


class _FakeDocumentLoader:
    """Return the current nested T01 document shape without external services."""

    def execute(self, request):
        return _FakeToolResult(
            status="success",
            data={
                "document": {
                    "document_ref": "storage://document/test",
                    "text": "Alice works for Acme Corporation.",
                    "confidence": 0.88,
                    "page_count": 1,
                    "extraction_method": "text",
                }
            },
        )


class _FakeTextChunker:
    """Capture T15 requests without external services."""

    def __init__(self) -> None:
        self.requests = []

    def execute(self, request):
        self.requests.append(request)
        return _FakeToolResult(
            status="success",
            data={
                "chunks": [{"chunk_ref": "chunk-1", "text": request.input_data["text"]}],
                "total_chunks": 1,
                "total_tokens": 5,
            },
        )


class _FakeEntityExtractor:
    """Return the current T23A entity output shape."""

    def execute(self, request):
        return _FakeToolResult(
            status="success",
            data={
                "entities": [
                    {
                        "surface_form": "Alice",
                        "entity_type": "PERSON",
                        "source_ref": request.input_data["chunk_ref"],
                        "start_pos": 0,
                        "end_pos": 5,
                    }
                ],
                "total_entities": 1,
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

    normalized = normalize_entities_for_t27(entities)

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

    normalized = normalize_entities_for_t27(entities)

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
        normalize_entities_for_t27([{"name": "Alice", "type": "PERSON"}])


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


@pytest.mark.asyncio
async def test_complete_pipeline_relationship_stage_sends_t27_entities() -> None:
    """CompleteGraphRAGPipeline should normalize mentions before direct T27 execution."""
    fake_extractor = _FakeRelationshipExtractor()
    pipeline = object.__new__(CompleteGraphRAGPipeline)
    pipeline.relationship_extractor = fake_extractor
    chunks = [{"chunk_ref": "chunk-1", "text": "Alice works at Acme Corp.", "confidence": 0.9}]
    mentions = [
        {
            "surface_form": "Alice",
            "entity_type": "PERSON",
            "source_ref": "chunk-1",
            "start_pos": 0,
            "end_pos": 5,
            "confidence": 0.91,
        },
        {
            "surface_form": "Acme Corp",
            "entity_type": "ORG",
            "source_ref": "chunk-1",
            "start_pos": 15,
            "end_pos": 24,
            "confidence": 0.93,
        },
    ]

    result = await pipeline._execute_relationship_extraction(chunks, mentions)

    assert result["status"] == "success"
    assert result["relationship_count"] == 1
    sent_entities = fake_extractor.requests[0].input_data["entities"]
    assert sent_entities[0]["text"] == "Alice"
    assert sent_entities[0]["start"] == 0
    assert sent_entities[1]["text"] == "Acme Corp"
    assert sent_entities[1]["end"] == 24


@pytest.mark.asyncio
async def test_complete_pipeline_relationship_stage_groups_current_t23a_chunk_refs() -> None:
    """Current T23A entities use chunk_ref, so complete-pipeline must group by it."""
    fake_extractor = _FakeRelationshipExtractor()
    pipeline = object.__new__(CompleteGraphRAGPipeline)
    pipeline.relationship_extractor = fake_extractor
    chunks = [{"chunk_ref": "chunk-1", "text": "Alice works at Acme Corp.", "confidence": 0.9}]
    mentions = [
        {
            "surface_form": "Alice",
            "entity_type": "PERSON",
            "chunk_ref": "chunk-1",
            "start_pos": 0,
            "end_pos": 5,
            "confidence": 0.91,
        },
        {
            "surface_form": "Acme Corp",
            "entity_type": "ORG",
            "chunk_ref": "chunk-1",
            "start_pos": 15,
            "end_pos": 24,
            "confidence": 0.93,
        },
    ]

    result = await pipeline._execute_relationship_extraction(chunks, mentions)

    assert result["status"] == "success"
    assert result["relationship_count"] == 1
    sent_input = fake_extractor.requests[0].input_data
    assert sent_input["chunk_ref"] == "chunk-1"
    assert [entity["text"] for entity in sent_input["entities"]] == ["Alice", "Acme Corp"]


@pytest.mark.asyncio
async def test_complete_pipeline_document_loading_reads_current_t01_shape() -> None:
    """CompleteGraphRAGPipeline should adapt the current nested T01 document output."""
    pipeline = object.__new__(CompleteGraphRAGPipeline)
    pipeline.pdf_loader = _FakeDocumentLoader()

    result = await pipeline._execute_document_loading("/tmp/sample.txt")

    assert result == {
        "status": "success",
        "text_content": "Alice works for Acme Corporation.",
        "document_ref": "storage://document/test",
        "confidence": 0.88,
        "pages_processed": 1,
        "processing_method": "text",
    }


@pytest.mark.asyncio
async def test_complete_pipeline_text_chunking_sends_current_t15_shape() -> None:
    """CompleteGraphRAGPipeline should adapt to the current T15 input/output contract."""
    fake_chunker = _FakeTextChunker()
    pipeline = object.__new__(CompleteGraphRAGPipeline)
    pipeline.text_chunker = fake_chunker

    result = await pipeline._execute_text_chunking(
        "storage://document/test",
        "Alice works for Acme Corporation.",
    )

    assert result["status"] == "success"
    assert result["chunk_count"] == 1
    assert result["total_tokens"] == 5
    sent_input = fake_chunker.requests[0].input_data
    assert sent_input == {
        "document_ref": "storage://document/test",
        "text": "Alice works for Acme Corporation.",
        "document_confidence": 0.9,
    }


@pytest.mark.asyncio
async def test_complete_pipeline_entity_stage_reads_current_t23a_shape() -> None:
    """CompleteGraphRAGPipeline should collect entities from current T23A output."""
    pipeline = object.__new__(CompleteGraphRAGPipeline)
    pipeline.ner_extractor = _FakeEntityExtractor()

    result = await pipeline._execute_entity_extraction(
        [{"chunk_ref": "chunk-1", "text": "Alice works for Acme.", "confidence": 0.9}]
    )

    assert result["status"] == "success"
    assert result["mention_count"] == 1
    assert result["mentions"][0]["surface_form"] == "Alice"
    assert result["entity_types"] == {"PERSON": 1}


def test_complete_pipeline_default_queries_use_extracted_entity_names() -> None:
    """Default query smoke tests should target entities that T49 can look up."""
    pipeline = object.__new__(CompleteGraphRAGPipeline)
    mentions = [
        {"text": "Alice", "entity_type": "PERSON"},
        {"text": "Acme Corporation", "entity_type": "ORG"},
        {"text": "Alice", "entity_type": "PERSON"},
        {"text": "Seattle", "entity_type": "GPE"},
    ]

    assert pipeline._build_default_test_queries(mentions) == [
        "Alice",
        "Acme Corporation",
        "Seattle",
    ]
