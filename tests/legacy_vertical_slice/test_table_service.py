"""Explicit legacy vertical-slice checks for the historical SQLite table service."""

import importlib.util
import json
from pathlib import Path


def _load_table_service_class():
    """Load the legacy module directly so old package path assumptions stay inert."""
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "tool_compatability/poc/vertical_slice/services/table_service.py"
    spec = importlib.util.spec_from_file_location("legacy_vertical_slice_table_service", module_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TableService


def test_table_service_persists_embeddings_and_data_in_tmp_database(tmp_path: Path) -> None:
    """The historical service should keep its basic SQLite contract when isolated."""
    table_service = _load_table_service_class()
    service = table_service(str(tmp_path / "vertical_slice.db"))

    embedding_id = service.save_embedding("test text", [1.0, 2.0, 3.0])
    data_id = service.save_data("test_key", {"value": 123})
    embeddings = service.get_embeddings(limit=1)

    assert embedding_id == 1
    assert data_id == 1
    assert len(embeddings) == 1
    assert embeddings[0]["text"] == "test text"
    assert json.loads(embeddings[0]["embedding"]) == [1.0, 2.0, 3.0]


def test_table_service_returns_newest_embeddings_first(tmp_path: Path) -> None:
    """Legacy callers rely on recent embeddings being returned in descending id order."""
    table_service = _load_table_service_class()
    service = table_service(str(tmp_path / "vertical_slice.db"))

    service.save_embedding("first", [1.0])
    service.save_embedding("second", [2.0])

    embeddings = service.get_embeddings(limit=2)

    assert [row["text"] for row in embeddings] == ["second", "first"]
