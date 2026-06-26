"""Explicit legacy vertical-slice checks for pure framework registry behavior."""

import importlib.util
from pathlib import Path


def _load_clean_framework_module():
    """Load the legacy framework module while avoiding script-style integration imports."""
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "tool_compatability/poc/vertical_slice/framework/clean_framework.py"
    spec = importlib.util.spec_from_file_location("legacy_vertical_slice_clean_framework", module_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _registry_only_framework(module):
    """Create the registry surface without opening Neo4j or SQLite-backed services."""
    framework = object.__new__(module.CleanToolFramework)
    framework.tools = {}
    framework.capabilities = {}
    return framework


def test_register_tool_records_tool_and_declared_capabilities() -> None:
    """Tool registration should preserve the tool object and its capability declaration."""
    module = _load_clean_framework_module()
    framework = _registry_only_framework(module)
    tool = object()
    capabilities = module.ToolCapabilities(
        tool_id="VectorTool",
        input_type=module.DataType.TEXT,
        output_type=module.DataType.VECTOR,
        input_construct="text",
        output_construct="embedding",
        transformation_type="embedding",
    )

    framework.register_tool(tool, capabilities)

    assert framework.tools == {"VectorTool": tool}
    assert framework.capabilities == {"VectorTool": capabilities}


def test_find_chain_returns_shortest_registered_capability_path() -> None:
    """The historical BFS planner should compose registered tools by data type."""
    module = _load_clean_framework_module()
    framework = _registry_only_framework(module)
    framework.register_tool(
        object(),
        module.ToolCapabilities(
            tool_id="VectorTool",
            input_type=module.DataType.TEXT,
            output_type=module.DataType.VECTOR,
            input_construct="text",
            output_construct="embedding",
            transformation_type="embedding",
        ),
    )
    framework.register_tool(
        object(),
        module.ToolCapabilities(
            tool_id="TableTool",
            input_type=module.DataType.VECTOR,
            output_type=module.DataType.TABLE,
            input_construct="embedding",
            output_construct="stored",
            transformation_type="persistence",
        ),
    )

    chain = framework.find_chain(module.DataType.TEXT, module.DataType.TABLE)

    assert chain == ["VectorTool", "TableTool"]


def test_find_chain_returns_none_when_no_registered_path_exists() -> None:
    """Unreachable type pairs should fail closed instead of inventing a chain."""
    module = _load_clean_framework_module()
    framework = _registry_only_framework(module)
    framework.register_tool(
        object(),
        module.ToolCapabilities(
            tool_id="VectorTool",
            input_type=module.DataType.TEXT,
            output_type=module.DataType.VECTOR,
            input_construct="text",
            output_construct="embedding",
            transformation_type="embedding",
        ),
    )

    chain = framework.find_chain(module.DataType.FILE, module.DataType.TABLE)

    assert chain is None
