---
type: SourceSummary
title: Digimon Lineage Root Cleanup 2025 08 29
description: Root cleanup archive preserving stale root artifacts, duplicate KGAS applications, specialized Twitter explorer app, Kubernetes deployment manifests, stale DB/config outputs, and cleanup rationale.
tags: [source, digimon-lineage, archive, root-cleanup, applications, k8s, entry-points, clutter]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/root_cleanup_2025_08_29/
confidence: high
---

# Summary

`archive/root_cleanup_2025_08_29/` is a 63-file, 1.7 MB root-cleanup archive. Its aggregate content-manifest hash is `9100576394b5822b29b56db7ced98f011b611b328ed9c9d77230bf52d8db2811`. [1]

The README says these files were archived during root directory cleanup to remove superseded inventories, stale output files, duplicate applications, Kubernetes deployment material, and a stale vertical-slice database. It explicitly says the `apps/` directory caused a "multiple personalities" or "4 entry points" problem that was resolved by archiving it. [2]

# Inventory

| Area | Examples | Role |
| --- | --- | --- |
| Root files | `README.md`, `IMPLEMENTATION_START.md`, `tool_inventory.json`, `cross_modal_final_test.log`, `performance_data.json`, `sla_config.json`, `vertical_slice.db` | Cleanup rationale, historical checkpoint, stale output/config/database artifacts, and tool inventory. [1] |
| `apps/kgas/` | `main.py`, `streamlit_app.py`, `kgas_mcp_server.py`, `kgas_simple_mcp_server.py`, integration scripts | Duplicate KGAS app entry points moved out of active root. [3] |
| `apps/specialized/twitterexplorer/` | Streamlit/Twitter analysis app, local `.git`, PyVis HTML, API/LLM/config modules | Specialized app archived with its nested repo metadata and generated graph HTML. [1] |
| `k8s/` | `deployment.yaml`, `service.yaml`, `configmap.yaml`, `secret.yaml`, `CLAUDE.md` | Enterprise Kubernetes deployment material not needed for thesis scope. [1] |

# Cleanup Rationale

The archive README states:

- `tool_inventory.json` was superseded by `combined_tool_inventory.json`. [2]
- `IMPLEMENTATION_START.md` was a historical 2025-08-26 checkpoint. [2]
- SLA/performance JSON and `vertical_slice.db` were stale or not thesis-needed. [2]
- `apps/` contained duplicate applications and duplicate entry points. [2]
- `k8s/` contained enterprise-grade deployment material not needed for the thesis. [2]

This is useful policy lineage: root cleanup was not just space-saving; it reduced agent confusion by removing duplicate app personalities from the active root.

# Historical Checkpoint

`IMPLEMENTATION_START.md` records the start of tool-composition framework integration on 2025-08-26. It says Phases 1-3 were complete, CLAUDE.md had comprehensive instructions, and the plan was to implement CompositionService, UniversalAdapterFactory, TextLoader integration, and evidence documentation, with real services only and under 20% performance overhead. [4]

# Archived Apps

`apps/kgas/README.md` describes KGAS application files and entry points, including Streamlit UI, main application, MCP server implementations, Claude Code integration examples, and UI integration. [3]

The specialized `twitterexplorer` app includes a nested `.git` directory, Streamlit app, RapidAPI/Gemini-oriented modules, graph manager, merged endpoint data, and a large generated PyVis HTML graph. This looks like a separate specialized exploratory app rather than a canonical KGAS entry point. [1]

# Cross-Modal Test Output

`cross_modal_final_test.log` records a cross-modal tool import test. It successfully registered GraphTableExporter, MultiFormatExporter, CrossModalTool, AsyncTextEmbedder, and CrossModalConverter, but failed T15B VectorEmbedderKGAS because `src.tools.phase1.t15b_vector_embedder_kgas` was missing. It also notes CrossModalConverter failed to initialize an embedding service because `torch` was missing, while still registering. [5]

# Security And Placeholder Secrets

A targeted scan found no literal OpenAI or Google API keys. The Kubernetes `secret.yaml` does contain base64-encoded placeholder values such as `secret_key_change_me`, `api_key_change_me`, and `production_password`. Treat these as placeholder-secret hygiene issues, not evidence of live credential leakage. [6]

# Interpretation

This archive explains an important cleanup decision: remove stale outputs and duplicate application surfaces from the active project root so agents and humans see a clearer canonical entry-point structure. It should be read alongside current-code verification pages when asking what the active entry point is now.

# Relationship To Wiki

- [Digimon Lineage Archive Coverage Audit 2026-06-25](digimon-lineage-archive-coverage-audit-2026-06-25.md): queue-control page that identified this archive area.
- [Digimon Lineage UI Recovered Components](digimon-lineage-ui-recovered-components.md): related UI/demo/app archive material.
- [Digimon Lineage Generated Outputs 2025 08](digimon-lineage-generated-outputs-2025-08.md): related stale output/config artifacts.
- [Current Code Verification 2026-06-25](current-code-verification-2026-06-25.md): current checkout entry-point/status verification.
- [Current Status Verification Discipline](../concepts/current-status-verification-discipline.md): relevant distinction between active entry points and archived duplicates.

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/root_cleanup_2025_08_29/`

[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/root_cleanup_2025_08_29/README.md`

[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/root_cleanup_2025_08_29/apps/kgas/README.md`

[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/root_cleanup_2025_08_29/IMPLEMENTATION_START.md`

[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/root_cleanup_2025_08_29/cross_modal_final_test.log`

[6] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/root_cleanup_2025_08_29/k8s/secret.yaml`
