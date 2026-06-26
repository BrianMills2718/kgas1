# Legacy Vertical Slice POC

This directory is historical proof-of-concept material. It is preserved because it
contains useful architecture and thesis evidence about tool chaining, service-tool
integration, Neo4j/SQLite coordination, uncertainty, provenance, and document
isolation.

It is **not** part of the default pytest suite.

Default pytest is intentionally scoped to `tests/current_runtime` in
`pyproject.toml`. Do not broaden default collection to this directory without a
separate migration plan.

## Why This Is Excluded

Several files in this directory are named `test*.py` but behave like executable
scripts. Some execute work at import time, including:

- loading historical absolute paths such as `/home/brian/projects/Digimons`;
- using stale Neo4j credentials such as `devpassword`;
- creating local SQLite files such as `vertical_slice.db`;
- running broad graph cleanup statements such as
  `MATCH (n:VSEntity) DETACH DELETE n`;
- depending on working-directory-specific fixture paths;
- calling provider-backed extraction paths.

Running broad pytest collection over this directory can trigger side effects
before pytest reaches the test body.

## Safe Handling

Safe:

- read files as historical sources;
- cite the architecture/audit documents;
- extract small no-IO tests into `tests/legacy_vertical_slice/`;
- use `tmp_path` for SQLite/file outputs in migrated tests;
- gate live Neo4j tests behind explicit environment variables and exact run
  scopes.

Unsafe by default:

- `pytest tool_compatability/poc/vertical_slice`;
- importing script-style `test*.py` files casually;
- executing anything that uses `devpassword`;
- broad Neo4j cleanup;
- moving or renaming historical files without a provenance manifest.

## Revival Plan

Use the plan before changing this directory:

`../../../investigations/2026-06-26-legacy-vertical-slice-test-revival-plan.md`

First recommended migration slice:

1. Extract `test_services.py` table-service checks into
   `tests/legacy_vertical_slice/test_table_service.py` using `tmp_path`.
2. Add a no-IO framework capability registration test based on
   `test_integration.py`.
3. Leave Neo4j and provider-dependent tests untouched until a live-harness plan
   exists.
