"""Runtime contract checks for the cross-modal API boundary."""

import os
import sys
from types import SimpleNamespace
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import pytest
from fastapi import HTTPException

from src.api import cross_modal_api as api
from src.analytics.cross_modal_service_registry import CrossModalServiceRegistry
from src.analytics.governed_llm_client_adapter import GovernedLLMClientAdapter
from src.analytics.mode_selection_service import AnalysisMode, ConfidenceLevel, ModeSelectionResult


def _tiny_pdf_bytes(text: str) -> bytes:
    """Build a tiny text-bearing PDF fixture without adding test dependencies."""
    objects = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\nendobj\n",
        b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
    ]
    stream = f"BT /F1 24 Tf 72 720 Td ({text}) Tj ET".encode()
    objects.append(
        b"5 0 obj\n<< /Length "
        + str(len(stream)).encode()
        + b" >>\nstream\n"
        + stream
        + b"\nendstream\nendobj\n"
    )
    content = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(content))
        content.extend(obj)
    xref_offset = len(content)
    content.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    content.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        content.extend(f"{offset:010d} 00000 n \n".encode())
    content.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode()
    )
    return bytes(content)


def _tiny_docx_bytes(text: str) -> bytes:
    """Build a tiny DOCX fixture in memory using the project dependency."""
    import docx

    buffer = BytesIO()
    document = docx.Document()
    document.add_paragraph(text)
    document.save(buffer)
    return buffer.getvalue()


def test_cross_modal_api_imports_without_registry_runtime_dependencies() -> None:
    """The API module should import before optional service-registry dependencies are installed."""
    assert api.app.title == "KGAS Cross-Modal Analysis API"


@pytest.mark.parametrize(
    ("enum_cls", "raw_value", "expected"),
    [
        (api.DataFormat, "GRAPH", api.DataFormat.GRAPH),
        (api.WorkflowOptimizationLevel, "standard", api.WorkflowOptimizationLevel.STANDARD),
        (api.ValidationLevel, "STANDARD", api.ValidationLevel.STANDARD),
    ],
)
def test_parse_enum_accepts_case_insensitive_values(enum_cls, raw_value, expected) -> None:
    """API query strings should match enum values independent of case."""
    assert api._parse_enum(enum_cls, raw_value, "field") == expected


def test_parse_enum_returns_http_400_for_bad_value() -> None:
    """Invalid API enum values should fail as client errors, not generic runtime errors."""
    with pytest.raises(HTTPException) as exc_info:
        api._parse_enum(api.DataFormat, "network", "target format")

    assert exc_info.value.status_code == 400
    assert "graph" in exc_info.value.detail


@pytest.mark.asyncio
async def test_analyze_document_runs_complete_pipeline_for_txt(monkeypatch) -> None:
    """Text uploads should run through the real complete-pipeline adapter."""
    calls = []

    class _FakePipeline:
        async def process_document(self, document_path):
            calls.append(document_path)
            assert Path(document_path).exists()
            assert Path(document_path).read_text() == "Alice works for Acme Corporation."
            return {
                "status": "success",
                "transaction_id": "tx-test",
                "pipeline_stats": {
                    "chunks_created": 1,
                    "entities_extracted": 2,
                    "relationships_extracted": 1,
                    "graph_nodes_created": 2,
                    "graph_edges_created": 1,
                    "queries_answered": 3,
                },
                "pipeline_results": {
                    "document_loading": {"document_ref": "storage://document/test"},
                    "relationship_extraction": {"relationship_count": 1},
                },
                "validation": {"pipeline_complete": True, "neo4j_verified": True},
                "proof_of_completion": {
                    "all_steps_executed": True,
                    "real_operations_confirmed": True,
                    "neo4j_integration_verified": True,
                    "end_to_end_success": True,
                },
            }

    monkeypatch.setattr(api, "_create_complete_pipeline", lambda: _FakePipeline())

    response = await api.analyze_document(
        background_tasks=None,
        file=api.UploadFile(file=BytesIO(b"Alice works for Acme Corporation."), filename="sample.txt"),
        target_format="graph",
        task="extract entities",
        optimization_level="standard",
        validation_level="standard",
    )

    assert calls
    assert not Path(calls[0]).exists()
    assert response["workflow_id"] == "tx-test"
    assert response["selected_mode"] == "complete_graphrag_pipeline"
    assert response["results"]["pipeline_stats"]["relationships_extracted"] == 1
    assert response["validation"]["pipeline_complete"] is True
    assert response["source_traceability"] == {
        "filename": "sample.txt",
        "file_type": ".txt",
        "document_ref": "storage://document/test",
    }


@pytest.mark.asyncio
async def test_analyze_document_runs_complete_pipeline_for_pdf(monkeypatch) -> None:
    """PDF uploads should preserve the real suffix and run through the complete-pipeline adapter."""
    calls = []

    class _FakePipeline:
        async def process_document(self, document_path):
            calls.append(document_path)
            assert Path(document_path).exists()
            assert Path(document_path).suffix == ".pdf"
            return {
                "status": "success",
                "transaction_id": "tx-pdf-test",
                "pipeline_stats": {
                    "chunks_created": 1,
                    "entities_extracted": 2,
                    "relationships_extracted": 1,
                    "graph_nodes_created": 2,
                    "graph_edges_created": 1,
                    "queries_answered": 1,
                },
                "pipeline_results": {
                    "document_loading": {"document_ref": "storage://document/pdf-test"},
                    "relationship_extraction": {"relationship_count": 1},
                },
                "validation": {"pipeline_complete": True, "neo4j_verified": True},
                "proof_of_completion": {
                    "all_steps_executed": True,
                    "real_operations_confirmed": True,
                    "neo4j_integration_verified": True,
                    "end_to_end_success": True,
                },
            }

    monkeypatch.setattr(api, "_create_complete_pipeline", lambda: _FakePipeline())

    response = await api.analyze_document(
        background_tasks=None,
        file=api.UploadFile(file=BytesIO(_tiny_pdf_bytes("Alice works for Acme.")), filename="sample.pdf"),
        target_format="graph",
        task="extract entities",
        optimization_level="standard",
        validation_level="standard",
    )

    assert calls
    assert not Path(calls[0]).exists()
    assert response["workflow_id"] == "tx-pdf-test"
    assert response["source_traceability"] == {
        "filename": "sample.pdf",
        "file_type": ".pdf",
        "document_ref": "storage://document/pdf-test",
    }


@pytest.mark.asyncio
async def test_analyze_document_runs_complete_pipeline_for_markdown(monkeypatch) -> None:
    """Markdown uploads should preserve the real suffix and run through the complete-pipeline adapter."""
    calls = []

    class _FakePipeline:
        async def process_document(self, document_path):
            calls.append(document_path)
            assert Path(document_path).exists()
            assert Path(document_path).suffix == ".md"
            assert Path(document_path).read_text() == "# Test\nAlice works for Acme.\n"
            return {
                "status": "success",
                "transaction_id": "tx-md-test",
                "pipeline_stats": {
                    "chunks_created": 1,
                    "entities_extracted": 2,
                    "relationships_extracted": 1,
                    "graph_nodes_created": 2,
                    "graph_edges_created": 1,
                    "queries_answered": 1,
                },
                "pipeline_results": {
                    "document_loading": {"document_ref": "storage://document/md-test"},
                    "relationship_extraction": {"relationship_count": 1},
                },
                "validation": {"pipeline_complete": True, "neo4j_verified": True},
                "proof_of_completion": {
                    "all_steps_executed": True,
                    "real_operations_confirmed": True,
                    "neo4j_integration_verified": True,
                    "end_to_end_success": True,
                },
            }

    monkeypatch.setattr(api, "_create_complete_pipeline", lambda: _FakePipeline())

    response = await api.analyze_document(
        background_tasks=None,
        file=api.UploadFile(file=BytesIO(b"# Test\nAlice works for Acme.\n"), filename="sample.md"),
        target_format="graph",
        task="extract entities",
        optimization_level="standard",
        validation_level="standard",
    )

    assert calls
    assert not Path(calls[0]).exists()
    assert response["workflow_id"] == "tx-md-test"
    assert response["source_traceability"] == {
        "filename": "sample.md",
        "file_type": ".md",
        "document_ref": "storage://document/md-test",
    }


@pytest.mark.asyncio
async def test_analyze_document_runs_complete_pipeline_for_docx(monkeypatch) -> None:
    """DOCX uploads should preserve the real suffix and run through the complete-pipeline adapter."""
    calls = []

    class _FakePipeline:
        async def process_document(self, document_path):
            calls.append(document_path)
            assert Path(document_path).exists()
            assert Path(document_path).suffix == ".docx"
            return {
                "status": "success",
                "transaction_id": "tx-docx-test",
                "pipeline_stats": {
                    "chunks_created": 1,
                    "entities_extracted": 2,
                    "relationships_extracted": 1,
                    "graph_nodes_created": 2,
                    "graph_edges_created": 1,
                    "queries_answered": 1,
                },
                "pipeline_results": {
                    "document_loading": {"document_ref": "storage://document/docx-test"},
                    "relationship_extraction": {"relationship_count": 1},
                },
                "validation": {"pipeline_complete": True, "neo4j_verified": True},
                "proof_of_completion": {
                    "all_steps_executed": True,
                    "real_operations_confirmed": True,
                    "neo4j_integration_verified": True,
                    "end_to_end_success": True,
                },
            }

    monkeypatch.setattr(api, "_create_complete_pipeline", lambda: _FakePipeline())

    response = await api.analyze_document(
        background_tasks=None,
        file=api.UploadFile(file=BytesIO(_tiny_docx_bytes("Alice works for Acme.")), filename="sample.docx"),
        target_format="graph",
        task="extract entities",
        optimization_level="standard",
        validation_level="standard",
    )

    assert calls
    assert not Path(calls[0]).exists()
    assert response["workflow_id"] == "tx-docx-test"
    assert response["source_traceability"] == {
        "filename": "sample.docx",
        "file_type": ".docx",
        "document_ref": "storage://document/docx-test",
    }


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("NEO4J_PASSWORD"), reason="requires local Neo4j credentials")
async def test_analyze_document_txt_upload_runs_live_complete_pipeline() -> None:
    """The API should expose the proven `.txt` complete-pipeline path when Neo4j is configured."""
    response = await api.analyze_document(
        background_tasks=None,
        file=api.UploadFile(
            file=BytesIO(b"Alice works for Acme Corporation. Bob founded Beta Labs in Seattle."),
            filename="sample.txt",
        ),
        target_format="graph",
        task="extract entities",
        optimization_level="standard",
        validation_level="standard",
    )

    stats = response["results"]["pipeline_stats"]
    proof = response["results"]["proof_of_completion"]
    assert response["selected_mode"] == "complete_graphrag_pipeline"
    assert stats["entities_extracted"] >= 4
    assert stats["relationships_extracted"] >= 1
    assert stats["graph_nodes_created"] >= 4
    assert stats["graph_edges_created"] >= 1
    assert stats["queries_answered"] >= 1
    assert proof["neo4j_integration_verified"] is True
    assert proof["end_to_end_success"] is True
    assert response["source_traceability"]["filename"] == "sample.txt"


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("NEO4J_PASSWORD"), reason="requires local Neo4j credentials")
async def test_analyze_document_pdf_upload_runs_live_complete_pipeline() -> None:
    """The API should expose a proven tiny `.pdf` complete-pipeline path when Neo4j is configured."""
    response = await api.analyze_document(
        background_tasks=None,
        file=api.UploadFile(
            file=BytesIO(_tiny_pdf_bytes("Alice works for Acme Corporation. Bob founded Beta Labs in Seattle.")),
            filename="sample.pdf",
        ),
        target_format="graph",
        task="extract entities",
        optimization_level="standard",
        validation_level="standard",
    )

    stats = response["results"]["pipeline_stats"]
    proof = response["results"]["proof_of_completion"]
    assert response["selected_mode"] == "complete_graphrag_pipeline"
    assert stats["entities_extracted"] >= 4
    assert stats["relationships_extracted"] >= 1
    assert stats["graph_nodes_created"] >= 4
    assert stats["graph_edges_created"] >= 1
    assert stats["queries_answered"] >= 1
    assert proof["neo4j_integration_verified"] is True
    assert proof["end_to_end_success"] is True
    assert response["source_traceability"]["filename"] == "sample.pdf"
    assert response["source_traceability"]["file_type"] == ".pdf"
    assert response["source_traceability"]["document_ref"].startswith("storage://document/")


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("NEO4J_PASSWORD"), reason="requires local Neo4j credentials")
async def test_analyze_document_markdown_upload_runs_live_complete_pipeline() -> None:
    """The API should expose a proven tiny `.md` complete-pipeline path when Neo4j is configured."""
    response = await api.analyze_document(
        background_tasks=None,
        file=api.UploadFile(
            file=BytesIO(b"# Test\nAlice works for Acme Corporation. Bob founded Beta Labs in Seattle.\n"),
            filename="sample.md",
        ),
        target_format="graph",
        task="extract entities",
        optimization_level="standard",
        validation_level="standard",
    )

    stats = response["results"]["pipeline_stats"]
    proof = response["results"]["proof_of_completion"]
    assert response["selected_mode"] == "complete_graphrag_pipeline"
    assert stats["entities_extracted"] >= 4
    assert stats["relationships_extracted"] >= 1
    assert stats["graph_nodes_created"] >= 4
    assert stats["graph_edges_created"] >= 1
    assert stats["queries_answered"] >= 1
    assert proof["neo4j_integration_verified"] is True
    assert proof["end_to_end_success"] is True
    assert response["source_traceability"]["filename"] == "sample.md"
    assert response["source_traceability"]["file_type"] == ".md"
    assert response["source_traceability"]["document_ref"].startswith("storage://document/")


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("NEO4J_PASSWORD"), reason="requires local Neo4j credentials")
async def test_analyze_document_docx_upload_runs_live_complete_pipeline() -> None:
    """The API should expose a proven tiny `.docx` complete-pipeline path when Neo4j is configured."""
    response = await api.analyze_document(
        background_tasks=None,
        file=api.UploadFile(
            file=BytesIO(_tiny_docx_bytes("Alice works for Acme Corporation. Bob founded Beta Labs in Seattle.")),
            filename="sample.docx",
        ),
        target_format="graph",
        task="extract entities",
        optimization_level="standard",
        validation_level="standard",
    )

    stats = response["results"]["pipeline_stats"]
    proof = response["results"]["proof_of_completion"]
    assert response["selected_mode"] == "complete_graphrag_pipeline"
    assert stats["entities_extracted"] >= 4
    assert stats["relationships_extracted"] >= 1
    assert stats["graph_nodes_created"] >= 4
    assert stats["graph_edges_created"] >= 1
    assert stats["queries_answered"] >= 1
    assert proof["neo4j_integration_verified"] is True
    assert proof["end_to_end_success"] is True
    assert response["source_traceability"]["filename"] == "sample.docx"
    assert response["source_traceability"]["file_type"] == ".docx"
    assert response["source_traceability"]["document_ref"].startswith("storage://document/")


@pytest.mark.asyncio
async def test_analyze_document_returns_explicit_501_for_unwired_document_formats(monkeypatch) -> None:
    """The analyze endpoint should not claim unproven legacy DOC upload support."""
    def fail_get_registry():
        raise AssertionError("registry should not be loaded")

    monkeypatch.setattr(api, "_get_registry", fail_get_registry)

    with pytest.raises(HTTPException) as exc_info:
        await api.analyze_document(
            background_tasks=None,
            file=api.UploadFile(file=BytesIO(b"doc"), filename="sample.doc"),
            target_format="graph",
            task="extract entities",
            optimization_level="standard",
            validation_level="standard",
        )

    assert exc_info.value.status_code == 501
    assert "only for .txt, .pdf, .md, and .docx" in exc_info.value.detail


@pytest.mark.asyncio
async def test_analyze_document_rejects_unsupported_file_type() -> None:
    """Unsupported uploads should still fail as client errors before pipeline dispatch."""
    with pytest.raises(HTTPException) as exc_info:
        await api.analyze_document(
            background_tasks=None,
            file=api.UploadFile(file=BytesIO(b"data"), filename="sample.exe"),
            target_format="graph",
            task="extract entities",
            optimization_level="standard",
            validation_level="standard",
        )

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_analyze_document_rejects_invalid_target_format() -> None:
    """Invalid analyze enum parameters should remain 400s, not 501s."""
    with pytest.raises(HTTPException) as exc_info:
        await api.analyze_document(
            background_tasks=None,
            file=api.UploadFile(file=BytesIO(b"data"), filename="sample.txt"),
            target_format="network",
            task="extract entities",
            optimization_level="standard",
            validation_level="standard",
        )

    assert exc_info.value.status_code == 400


def test_preferred_modes_follow_requested_target_format() -> None:
    """The API should pass preferred modes into the current orchestrator signature."""
    preferred = api._preferred_modes_for_format(api.DataFormat.TABLE)

    assert preferred is not None
    assert [mode.value for mode in preferred] == ["table_analysis"]


@dataclass
class _FakeResult:
    analysis_metadata: dict


def test_selected_mode_value_handles_current_analysis_result_metadata() -> None:
    """The API response should read selected mode from the current AnalysisResult metadata."""
    result = _FakeResult(analysis_metadata={"mode_selection": {"primary_mode": "graph_analysis"}})

    assert api._selected_mode_value(result) == "graph_analysis"


def test_recommend_request_builds_current_data_context() -> None:
    """Recommendation requests should map into the current DataContext dataclass."""
    request = api.RecommendRequest(
        task="find central actors",
        data_type="graph",
        size=120,
        performance_priority="quality",
    )

    context = api._recommend_request_to_data_context(request)

    assert context.data_size == 120
    assert context.data_types == ["graph"]
    assert context.entity_count == 0
    assert context.relationship_count == 0
    assert context.has_hierarchical_structure is True
    assert context.available_formats == ["graph", "table", "vector"]


def test_recommend_request_rejects_negative_size() -> None:
    """Invalid recommendation sizes should be client errors."""
    request = api.RecommendRequest(
        task="find central actors",
        data_type="graph",
        size=-1,
        performance_priority="quality",
    )

    with pytest.raises(HTTPException) as exc_info:
        api._recommend_request_to_data_context(request)

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_startup_defaults_recommendation_llm_to_governed_client(monkeypatch) -> None:
    """API startup should default recommendation LLM calls to llm_client gpt-5.4-mini."""
    captured = {}

    def fake_initialize(config):
        captured.update(config)
        return True

    monkeypatch.setattr(api, "_initialize_cross_modal_services", fake_initialize)
    monkeypatch.delenv("KGAS_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("KGAS_RECOMMEND_MODEL", raising=False)
    monkeypatch.delenv("KGAS_RECOMMEND_MAX_BUDGET", raising=False)

    await api.startup_event()

    assert captured["llm"] == {
        "provider": "llm_client",
        "model": "gpt-5.4-mini",
        "max_budget": 5.0,
    }


def test_registry_builds_governed_llm_client_adapter() -> None:
    """The service registry should support the llm_client provider explicitly."""
    registry = CrossModalServiceRegistry()
    registry.shutdown()

    llm_service = registry._initialize_llm_service(
        {"provider": "llm_client", "model": "gpt-5.4-mini", "max_budget": 7.5}
    )

    assert isinstance(llm_service, GovernedLLMClientAdapter)
    assert llm_service.model == "gpt-5.4-mini"
    assert llm_service.max_budget == 7.5
    assert registry.llm_service is llm_service
    registry.shutdown()


@pytest.mark.asyncio
async def test_governed_llm_client_adapter_uses_required_observability_kwargs(monkeypatch) -> None:
    """The adapter should call llm_client with task, trace_id, and max_budget."""
    calls = []

    async def fake_acall_llm(model, messages, **kwargs):
        calls.append((model, messages, kwargs))
        return SimpleNamespace(
            content='{"primary_mode":"graph_analysis","secondary_modes":[],"confidence":0.91,"reasoning":"Graph fits.","key_factors":["graph"]}',
            model=model,
            requested_model=model,
            resolved_model="openrouter/openai/gpt-5.4-mini",
            cost=0.01,
            usage={"total_tokens": 42},
        )

    monkeypatch.setitem(sys.modules, "llm_client", SimpleNamespace(acall_llm=fake_acall_llm))
    adapter = GovernedLLMClientAdapter(model="gpt-5.4-mini", max_budget=3.0)

    response = await adapter.complete("select a mode")

    assert "graph_analysis" in response
    assert calls[0][0] == "gpt-5.4-mini"
    assert calls[0][1] == [{"role": "user", "content": "select a mode"}]
    assert calls[0][2]["task"] == "kgas.mode_selection"
    assert calls[0][2]["trace_id"].startswith("kgas.mode_selection:")
    assert calls[0][2]["max_budget"] == 3.0
    assert adapter.last_call_metadata["cost"] == 0.01


def test_serialize_mode_selection_uses_json_friendly_values() -> None:
    """Mode-selection dataclasses should serialize enum values for the API response."""
    result = ModeSelectionResult(
        primary_mode=AnalysisMode.GRAPH_ANALYSIS,
        secondary_modes=[AnalysisMode.TABLE_ANALYSIS],
        confidence=0.82,
        confidence_level=ConfidenceLevel.HIGH,
        reasoning="Graph structure is central to the task.",
        workflow_steps=[{"step": "graph"}],
        estimated_performance={"duration": 10},
        fallback_used=False,
        selection_metadata={"source": "fake"},
    )

    serialized = api._serialize_mode_selection(result)

    assert serialized["recommended_mode"] == "graph_analysis"
    assert serialized["secondary_modes"] == ["table_analysis"]
    assert serialized["confidence_level"] == "high"


@pytest.mark.asyncio
async def test_recommend_mode_calls_current_mode_selector_contract(monkeypatch) -> None:
    """The endpoint should call select_optimal_mode with research_question and DataContext."""
    calls = []

    class _FakeModeSelector:
        async def select_optimal_mode(self, research_question, data_context, preferences=None):
            calls.append((research_question, data_context, preferences))
            return ModeSelectionResult(
                primary_mode=AnalysisMode.GRAPH_ANALYSIS,
                secondary_modes=[],
                confidence=0.91,
                confidence_level=ConfidenceLevel.VERY_HIGH,
                reasoning="Graph analysis fits graph data.",
                workflow_steps=[],
                estimated_performance={},
                fallback_used=False,
                selection_metadata={},
            )

    class _FakeRegistry:
        mode_selector = _FakeModeSelector()

    monkeypatch.setattr(api, "_get_registry", lambda: _FakeRegistry())
    request = api.RecommendRequest(
        task="find central actors",
        data_type="graph",
        size=100,
        performance_priority="speed",
    )

    response = await api.recommend_mode(request)

    assert response["recommended_mode"] == "graph_analysis"
    assert calls[0][0] == "find central actors"
    assert calls[0][1].data_types == ["graph"]
    assert calls[0][2] == {"performance_priority": "speed"}


@dataclass
class _FakeConversionMetadata:
    conversion_timestamp: str = "2026-06-25T00:00:00"
    processing_time: float = 0.25
    data_size_before: int = 12
    data_size_after: int = 20
    semantic_features_preserved: list = None
    quality_metrics: dict = None
    conversion_parameters: dict = None


@dataclass
class _FakeConversionResult:
    data: dict
    source_format: api.DataFormat
    target_format: api.DataFormat
    preservation_score: float
    conversion_metadata: _FakeConversionMetadata
    validation_passed: bool
    semantic_integrity: bool
    warnings: list


@pytest.mark.asyncio
async def test_convert_format_calls_current_converter_contract(monkeypatch) -> None:
    """The convert endpoint should call convert_data and serialize the current result shape."""
    calls = []

    class _FakeConverter:
        async def convert_data(self, data, source_format, target_format, method=None):
            calls.append((data, source_format, target_format, method))
            return _FakeConversionResult(
                data={"rows": [{"id": "n1"}]},
                source_format=source_format,
                target_format=target_format,
                preservation_score=0.96,
                conversion_metadata=_FakeConversionMetadata(
                    semantic_features_preserved=["entities"],
                    quality_metrics={"size_preservation_ratio": 1.6},
                    conversion_parameters={"method": method},
                ),
                validation_passed=True,
                semantic_integrity=True,
                warnings=[],
            )

    class _FakeRegistry:
        converter = _FakeConverter()

    monkeypatch.setattr(api, "_get_registry", lambda: _FakeRegistry())
    request = api.ConvertRequest(
        data={"nodes": [{"id": "n1"}], "edges": []},
        source_format="graph",
        target_format="table",
        method="lossless",
    )

    response = await api.convert_format(request)

    assert calls == [
        (
            {"nodes": [{"id": "n1"}], "edges": []},
            api.DataFormat.GRAPH,
            api.DataFormat.TABLE,
            "lossless",
        )
    ]
    assert response["data"] == {"rows": [{"id": "n1"}]}
    assert response["source_format"] == "graph"
    assert response["target_format"] == "table"
    assert response["metadata"]["preservation_score"] == 0.96
    assert response["metadata"]["conversion_parameters"] == {"method": "lossless"}
    assert response["performance"]["conversion_time"] == 0.25
    assert response["performance"]["data_size_after"] == 20


@pytest.mark.asyncio
async def test_convert_format_returns_503_when_converter_missing(monkeypatch) -> None:
    """A missing converter is an unavailable service, not an internal server error."""
    class _FakeRegistry:
        converter = None

    monkeypatch.setattr(api, "_get_registry", lambda: _FakeRegistry())

    with pytest.raises(HTTPException) as exc_info:
        await api.convert_format(
            api.ConvertRequest(
                data={"nodes": [], "edges": []},
                source_format="graph",
                target_format="table",
            )
        )

    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_convert_format_same_source_and_target_short_circuits(monkeypatch) -> None:
    """No converter should be required when the requested formats already match."""
    def fail_get_registry():
        raise AssertionError("registry should not be loaded")

    monkeypatch.setattr(api, "_get_registry", fail_get_registry)

    response = await api.convert_format(
        api.ConvertRequest(
            data={"nodes": [], "edges": []},
            source_format="graph",
            target_format="GRAPH",
        )
    )

    assert response == {"data": {"nodes": [], "edges": []}, "message": "Source and target formats are the same"}


@pytest.mark.asyncio
async def test_get_statistics_preserves_registry_unavailable_status(monkeypatch) -> None:
    """Stats should not mask registry import failures as generic 500s."""
    def unavailable_registry():
        raise HTTPException(status_code=503, detail="registry unavailable")

    monkeypatch.setattr(api, "_get_registry", unavailable_registry)

    with pytest.raises(HTTPException) as exc_info:
        await api.get_statistics()

    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_batch_analyze_processes_files_through_complete_pipeline(monkeypatch) -> None:
    """Batch analysis should compose the proven single-document pipeline path."""
    calls = []

    class _FakePipeline:
        async def process_document(self, document_path):
            calls.append(Path(document_path).suffix)
            return {
                "status": "success",
                "transaction_id": f"tx-{len(calls)}",
                "pipeline_stats": {
                    "chunks_created": 1,
                    "entities_extracted": 2,
                    "relationships_extracted": 1,
                    "graph_nodes_created": 2,
                    "graph_edges_created": 1,
                    "queries_answered": 1,
                },
                "pipeline_results": {
                    "document_loading": {"document_ref": f"storage://document/batch-{len(calls)}"},
                    "relationship_extraction": {"relationship_count": 1},
                },
                "validation": {"pipeline_complete": True, "neo4j_verified": True},
                "proof_of_completion": {
                    "all_steps_executed": True,
                    "real_operations_confirmed": True,
                    "neo4j_integration_verified": True,
                    "end_to_end_success": True,
                },
            }

    monkeypatch.setattr(api, "_create_complete_pipeline", lambda: _FakePipeline())

    response = await api.batch_analyze(
        background_tasks=None,
        files=[
            api.UploadFile(file=BytesIO(b"Alice works for Acme."), filename="one.txt"),
            api.UploadFile(file=BytesIO(_tiny_pdf_bytes("Bob founded Beta Labs.")), filename="two.pdf"),
        ],
        target_format="graph",
        task="extract entities",
    )

    job = api.jobs[response.job_id]
    try:
        assert response.status == "completed"
        assert job["processed_files"] == 2
        assert [result["filename"] for result in job["results"]] == ["one.txt", "two.pdf"]
        assert job["errors"] == []
        assert calls == [".txt", ".pdf"]
    finally:
        api.jobs.pop(response.job_id, None)


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("NEO4J_PASSWORD"), reason="requires local Neo4j credentials")
async def test_batch_analyze_runs_live_complete_pipeline_for_txt_upload() -> None:
    """Batch analysis should run at least one upload through the live complete pipeline."""
    response = await api.batch_analyze(
        background_tasks=None,
        files=[
            api.UploadFile(
                file=BytesIO(b"Alice works for Acme Corporation. Bob founded Beta Labs in Seattle."),
                filename="batch.txt",
            )
        ],
        target_format="graph",
        task="extract entities",
    )

    job = api.jobs[response.job_id]
    try:
        assert job["status"] == "completed"
        assert job["processed_files"] == 1
        assert not job["errors"]
        result = job["results"][0]["result"]
        assert result["results"]["proof_of_completion"]["neo4j_integration_verified"] is True
        assert result["results"]["proof_of_completion"]["end_to_end_success"] is True
    finally:
        api.jobs.pop(response.job_id, None)


@pytest.mark.asyncio
async def test_batch_analyze_records_per_file_errors(monkeypatch) -> None:
    """Unsupported files should become per-file errors without aborting the whole batch."""
    class _FakePipeline:
        async def process_document(self, document_path):
            return {
                "status": "success",
                "transaction_id": "tx-ok",
                "pipeline_stats": {},
                "pipeline_results": {"document_loading": {"document_ref": "storage://document/ok"}},
                "validation": {},
                "proof_of_completion": {},
            }

    monkeypatch.setattr(api, "_create_complete_pipeline", lambda: _FakePipeline())

    response = await api.batch_analyze(
        background_tasks=None,
        files=[
            api.UploadFile(file=BytesIO(b"Alice works for Acme."), filename="one.txt"),
            api.UploadFile(file=BytesIO(b"legacy"), filename="legacy.doc"),
        ],
        target_format="graph",
        task="extract entities",
    )

    job = api.jobs[response.job_id]
    try:
        assert job["status"] == "completed"
        assert job["processed_files"] == 2
        assert len(job["results"]) == 1
        assert job["errors"][0]["filename"] == "legacy.doc"
        assert job["errors"][0]["status_code"] == 501
    finally:
        api.jobs.pop(response.job_id, None)


@pytest.mark.asyncio
async def test_process_batch_analysis_status_for_all_failed_files() -> None:
    """A batch with only unsupported files should be marked failed and create no demo results."""
    api.jobs["job-test"] = {
        "id": "job-test",
        "status": "pending",
        "created_at": "2026-06-25T00:00:00",
        "total_files": 0,
        "processed_files": 0,
        "results": [],
        "errors": [],
    }

    try:
        await api.process_batch_analysis(
            "job-test",
            [{"filename": "legacy.doc", "content": b"legacy"}],
            "graph",
            "extract entities",
        )

        assert api.jobs["job-test"]["status"] == "failed"
        assert api.jobs["job-test"]["results"] == []
        assert api.jobs["job-test"]["errors"][0]["status_code"] == 501
        assert "only for .txt" in api.jobs["job-test"]["errors"][0]["error"]
    finally:
        api.jobs.pop("job-test", None)


@pytest.mark.asyncio
async def test_get_job_status_returns_completed_batch_results(monkeypatch) -> None:
    """Job status should expose completed batch results and progress."""
    class _FakePipeline:
        async def process_document(self, document_path):
            return {
                "status": "success",
                "transaction_id": "tx-status",
                "pipeline_stats": {},
                "pipeline_results": {"document_loading": {"document_ref": "storage://document/status"}},
                "validation": {},
                "proof_of_completion": {},
            }

    monkeypatch.setattr(api, "_create_complete_pipeline", lambda: _FakePipeline())

    response = await api.batch_analyze(
        background_tasks=None,
        files=[api.UploadFile(file=BytesIO(b"Alice works for Acme."), filename="one.txt")],
        target_format="graph",
        task="extract entities",
    )

    try:
        status = await api.get_job_status(response.job_id)
        assert status["status"] == "completed"
        assert status["progress"]["percentage"] == 100
        assert status["results"][0]["filename"] == "one.txt"
    finally:
        api.jobs.pop(response.job_id, None)
