---
type: Source
title: Digimon Autoloop Operator Routing Variant
description: Source summary for the autoloop lineage variant's later operators-first DIGIMON state.
tags: [source, digimon, autoloop, operator-routing, mcp]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_autoloop/README.md
  - ../archive_full_record/lineage_variants/digimon_autoloop/CLAUDE.md
  - ../archive_full_record/lineage_variants/digimon_autoloop/FUNCTIONALITY.md
  - ../archive_full_record/lineage_variants/digimon_autoloop/docs/ACTIVE_DOCS.md
  - ../archive_full_record/lineage_variants/digimon_autoloop/docs/SYSTEM_OVERVIEW.md
  - ../archive_full_record/lineage_variants/digimon_autoloop/docs/COMPETITIVE_ANALYSIS.md
  - ../archive_full_record/lineage_variants/digimon_autoloop/docs/plans/03_prove_adaptive_routing.md
  - ../archive_full_record/lineage_variants/digimon_autoloop/docs/reports/2026-03-19_musique_50q_postmortem.md
confidence: medium
---

# Summary

`digimon_autoloop` preserves a later DIGIMON state where the system is framed as composable GraphRAG for multi-hop QA: 28 typed operators, agent-composed retrieval DAGs, MCP/direct Python interfaces, and explicit connection to JayLZhou GraphRAG. [1][2]

This is a major conceptual shift away from broad product surfaces. `FUNCTIONALITY.md` says the supported workflow is graph build, agent-driven retrieval, and grounded QA; it explicitly excludes maintained REST, Streamlit, social-media UI, and polished end-user document chat surfaces. [3]

# Current-Doc Boundary

`docs/ACTIVE_DOCS.md` is important because it separates current supported documentation from historical material. It declares README, FUNCTIONALITY, SYSTEM_OVERVIEW, GRAPH_ATTRIBUTE_MODEL, TOOL_CAPABILITY_MATRIX, COMPETITIVE_ANALYSIS, plan index, and `03_prove_adaptive_routing` as canonical, while older UI/API/UKRF/social-media/master-orchestrator roadmaps are historical unless rewritten. [4]

# Architecture Evidence

The autoloop docs define:

- 28 operators across entity, relationship, chunk, subgraph, community, and meta categories. [1][2][5]
- A uniform async operator signature and machine-readable operator registry. [2]
- MCP tools plus direct Python benchmark mode. [1][2]
- Two-model design: cheap graph-build model and configurable query-time reasoning model. [1][2][5]
- Graph storage using NetworkX/GraphML, FAISS vector indexes, and llm_client observability. [2][5]

# Benchmark Evidence

The same variant contains both strong-looking subset results and later negative development evidence. Earlier 50-question subset results report high HotpotQA and MuSiQue scores, but with explicit caveats that they are not comparable to full 1000-question SOTA benchmarks. [1][5][6]

The later controlled 50-question MuSiQue development comparison is negative for the adaptive-routing thesis:

| Mode | EM | LLM judge | Cost |
| --- | --- | --- | --- |
| baseline | 34.0% | 60.0% | $2.03 |
| fixed graph | 32.0% | 54.0% | $1.85 |
| hybrid/adaptive | 32.0% | 44.0% | $5.50 |

The active proof plan therefore classifies the work as "negative dev evidence, locked-eval decision pending," and the postmortem recommends one narrow salvage pass rather than treating the thesis as proven. [7][8]

# Interpretation

For thesis reconstruction, this variant is valuable because it captures the research question in its sharper form: whether adaptive operator routing over graph retrieval is worth the complexity. It is also a cautionary source: later docs were unusually explicit that current evidence did not yet support the adaptive-routing thesis.

# Citations

[1] `../archive_full_record/lineage_variants/digimon_autoloop/README.md`  
[2] `../archive_full_record/lineage_variants/digimon_autoloop/CLAUDE.md`  
[3] `../archive_full_record/lineage_variants/digimon_autoloop/FUNCTIONALITY.md`  
[4] `../archive_full_record/lineage_variants/digimon_autoloop/docs/ACTIVE_DOCS.md`  
[5] `../archive_full_record/lineage_variants/digimon_autoloop/docs/SYSTEM_OVERVIEW.md`  
[6] `../archive_full_record/lineage_variants/digimon_autoloop/docs/COMPETITIVE_ANALYSIS.md`  
[7] `../archive_full_record/lineage_variants/digimon_autoloop/docs/plans/03_prove_adaptive_routing.md`  
[8] `../archive_full_record/lineage_variants/digimon_autoloop/docs/reports/2026-03-19_musique_50q_postmortem.md`
