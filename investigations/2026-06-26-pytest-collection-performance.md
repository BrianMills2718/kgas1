---
title: Pytest Collection Performance Investigation
date: 2026-06-26
status: active
scope: phd_thesis_work broad pytest collection
---

# Question

Why does broad `pytest --collect-only -q` fail to produce output within roughly 90 seconds in `phd_thesis_work`, while `tests/current_runtime` runs quickly?

# Atoms

| ID | Question | Dependencies | Status |
| --- | --- | --- | --- |
| A1 | What pytest configuration controls collection scope and plugins? | none | answered |
| A2 | Which tests are collected under `tests/current_runtime`, and are there hidden tests elsewhere? | A1 | answered |
| A3 | Does collection hang during test module import, plugin initialization, or test discovery? | A1 | answered |
| A4 | Which import or fixture causes the delay if collection is import-bound? | A3 | answered |
| A5 | Is there an unambiguous safe fix, or should broad collection remain documented as a separate maintenance issue? | A1-A4 | answered |

# Assumptions Register

| # | Assumption | Confidence | How to verify | Round | Status |
| --- | --- | --- | --- | --- | --- |
| 1 | Broad collection is slow because pytest searches outside `tests/current_runtime`. | high | inspect pytest config and collection roots | 1 | Supported |
| 2 | The delay is caused by import side effects rather than test enumeration count. | high | collect/import test modules one by one with timing | 1 | Supported |
| 3 | A config-only fix may be possible if pytest is walking unintended directories. | high | inspect config and run scoped collection tests | 1 | Supported |

# Evidence

## Prior Observation

- `tests/current_runtime` passed quickly: `70 passed, 6 skipped`.
- With local Neo4j credentials loaded, `tests/current_runtime` passed live: `76 passed`, with non-failing Neo4j multi-record warnings.
- Broad `pytest --collect-only -q` produced no output after roughly 90 seconds and required terminating the local collection process.

## A1: Pytest Config

Before the fix, `pyproject.toml` had project/build/ruff configuration but no pytest collection configuration. There was no `pytest.ini`, `tox.ini`, `setup.cfg`, or `conftest.py` in the top-level scan.

Answer: default pytest collection was unconstrained.

## A2: Collection Surface

`tests/current_runtime` contains 15 test files and collects 76 tests in 2.82 seconds:

```text
76 tests collected in 2.82s
```

A repository scan excluding `.git`, `.venv`, `archive_full_record`, `thesis_record_wiki`, and `__pycache__` found additional `test*.py` files under `tool_compatability/poc/vertical_slice/`.

Answer: broad pytest was not equivalent to `tests/current_runtime`; it also considered legacy vertical-slice POC scripts.

## A3-A4: Import-Time Side Effects

Several `tool_compatability/poc/vertical_slice/test*.py` files are executable scripts, not safe pytest modules. They run work at import time:

- load an absolute historical env file such as `/home/brian/projects/Digimons/.env`;
- append absolute historical paths to `sys.path`;
- instantiate `CleanToolFramework`;
- connect to Neo4j with stale credentials;
- run graph cleanup like `MATCH (n:VSEntity) DETACH DELETE n`;
- create temporary working files in the current directory.

Targeted collection confirmed the failure mode:

```text
pytest --collect-only -q tests/current_runtime
# 76 tests collected in 2.82s

pytest --collect-only -q tool_compatability/poc/vertical_slice/test_5_tools.py
# collection error: Neo4j AuthError during module import

pytest --collect-only -q tool_compatability/poc/vertical_slice
# 11 tests collected, 6 collection errors
```

The vertical-slice collection errors included Neo4j `AuthError`, authentication rate limiting after repeated stale-password attempts, and `FileNotFoundError` for `ground_truth_paired/metadata/doc_001_clean.json` in `thesis_evidence/quick_paired_test.py`.

Answer: broad collection was blocked by legacy script-style tests with import-time side effects, not by current-runtime tests.

## A5: Fix

The safe fix is to configure pytest's default collection root to the maintained runtime test surface:

```toml
[tool.pytest.ini_options]
testpaths = ["tests/current_runtime"]
```

This preserves the legacy vertical-slice files as historical/POC artifacts while preventing default pytest from importing scripts that can attempt stale-auth Neo4j access or graph cleanup during collection.

# Findings

Root cause: unconstrained pytest discovery collected legacy vertical-slice POC scripts outside `tests/current_runtime`. Those files are named `test*.py` but contain top-level executable code with hardcoded historical paths, stale Neo4j credentials, graph cleanup statements, and missing relative data dependencies. Collection imported them and triggered side effects before pytest could safely enumerate tests.

Impact: broad `pytest --collect-only -q` was not a safe cheap gate. It could hang, error, or touch local Neo4j state depending on environment and authentication state.

Fix applied: `pyproject.toml` now sets pytest's default `testpaths` to `tests/current_runtime`.

Verification after fix:

```text
pytest --collect-only -q
# 76 tests collected in 2.64s

pytest -q
# 70 passed, 6 skipped in 3.45s
```

# Recommendations

1. Keep default pytest collection scoped to `tests/current_runtime`.
2. Treat `tool_compatability/poc/vertical_slice/test*.py` as historical POC scripts unless they are explicitly migrated into safe pytest tests.
3. If vertical-slice tests are revived, first move import-time execution under `if __name__ == "__main__"` or proper test functions, replace hardcoded `/home/brian/projects/Digimons` paths, remove import-time graph cleanup, and use explicit Neo4j credential gates.
4. Do not execute broad legacy collection against a shared Neo4j graph without isolation or an explicit backup/cleanup plan.
