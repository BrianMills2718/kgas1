"""Runtime contract checks for the cross-modal API boundary."""

from dataclasses import dataclass
from io import BytesIO

import pytest
from fastapi import HTTPException

from src.api import cross_modal_api as api
from src.analytics.mode_selection_service import AnalysisMode, ConfidenceLevel, ModeSelectionResult


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
async def test_analyze_document_returns_explicit_501_until_real_pipeline_wired(monkeypatch) -> None:
    """The analyze endpoint should not run metadata-only placeholder analysis."""
    def fail_get_registry():
        raise AssertionError("registry should not be loaded")

    monkeypatch.setattr(api, "_get_registry", fail_get_registry)

    with pytest.raises(HTTPException) as exc_info:
        await api.analyze_document(
            background_tasks=None,
            file=api.UploadFile(file=BytesIO(b"real text"), filename="sample.txt"),
            target_format="graph",
            task="extract entities",
            optimization_level="standard",
            validation_level="standard",
        )

    assert exc_info.value.status_code == 501
    assert "not wired" in exc_info.value.detail


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
async def test_batch_analyze_returns_explicit_501() -> None:
    """Batch analysis should fail honestly until wired to the current pipeline."""
    with pytest.raises(HTTPException) as exc_info:
        await api.batch_analyze(
            background_tasks=None,
            files=[],
            target_format="graph",
            task="extract entities",
        )

    assert exc_info.value.status_code == 501
    assert "not wired" in exc_info.value.detail


@pytest.mark.asyncio
async def test_process_batch_analysis_does_not_create_demo_results() -> None:
    """The background helper should not emit mock entities or relationships."""
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
        await api.process_batch_analysis("job-test", [], "graph", "extract entities")

        assert api.jobs["job-test"]["status"] == "failed"
        assert api.jobs["job-test"]["results"] == []
        assert "not wired" in api.jobs["job-test"]["errors"][0]["error"]
    finally:
        api.jobs.pop("job-test", None)
