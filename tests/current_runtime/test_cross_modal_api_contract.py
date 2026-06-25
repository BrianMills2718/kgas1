"""Runtime contract checks for the cross-modal API boundary."""

from dataclasses import dataclass

import pytest
from fastapi import HTTPException

from src.api import cross_modal_api as api


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


def test_document_placeholder_graph_preserves_document_metadata() -> None:
    """The current analyze endpoint sends explicit document metadata into the orchestrator."""
    graph = api._document_placeholder_graph("sample.txt", 42, ".txt")

    assert graph == {
        "nodes": [
            {
                "id": "document",
                "label": "sample.txt",
                "type": "DOCUMENT",
                "properties": {
                    "filename": "sample.txt",
                    "extension": ".txt",
                    "byte_count": 42,
                },
            }
        ],
        "edges": [],
    }


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
