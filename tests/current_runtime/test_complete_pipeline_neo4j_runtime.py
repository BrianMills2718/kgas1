"""Neo4j-backed runtime smoke test for the complete text pipeline."""

import os
import tempfile
import uuid
from pathlib import Path

import pytest

from src.analytics.complete_pipeline import CompleteGraphRAGPipeline


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("NEO4J_PASSWORD"), reason="requires local Neo4j credentials")
async def test_complete_pipeline_processes_tiny_txt_with_neo4j() -> None:
    """A source-scoped text file should execute real graph stages without relying on old smoke data."""
    run_marker = f"Scope{uuid.uuid4().hex[:8]}"
    document_text = (
        f"Helena {run_marker} works for Acme {run_marker} Corporation. "
        f"Marcus {run_marker} founded Beta {run_marker} Labs in Seattle {run_marker}."
    )
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as temp_file:
        temp_file.write(document_text)
        document_path = temp_file.name

    try:
        pipeline = CompleteGraphRAGPipeline()
        result = await pipeline.process_document(document_path)
    finally:
        Path(document_path).unlink(missing_ok=True)

    assert result["status"] == "success"
    assert result["pipeline_stats"]["chunks_created"] == 1
    assert result["pipeline_stats"]["entities_extracted"] >= 4
    assert result["pipeline_stats"]["relationships_extracted"] >= 1
    assert result["pipeline_stats"]["graph_nodes_created"] >= 4
    assert result["pipeline_stats"]["graph_edges_created"] >= 1
    assert result["pipeline_stats"]["queries_answered"] >= 1
    assert result["proof_of_completion"]["all_steps_executed"] is True
    assert result["proof_of_completion"]["real_operations_confirmed"] is True
    assert result["proof_of_completion"]["neo4j_integration_verified"] is True
    assert result["proof_of_completion"]["end_to_end_success"] is True

    pipeline_results = result["pipeline_results"]
    document_ref = pipeline_results["document_loading"]["document_ref"]
    graph_entities = pipeline_results["graph_building"]["entities"]
    graph_edges = pipeline_results["graph_building"]["edges"]
    assert all(document_ref in entity["properties"].get("source_refs", []) for entity in graph_entities)
    assert all(document_ref in edge["properties"].get("source_refs", []) for edge in graph_edges)
