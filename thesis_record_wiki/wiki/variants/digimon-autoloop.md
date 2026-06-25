---
type: LineageVariant
title: Digimon Autoloop
description: Autoloop-related lineage variant.
tags: [variant, digimon, autoloop]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/metadata/recovery_inventory.tsv
confidence: medium
---

# Summary

`digimon_autoloop` is preserved at `../archive_full_record/lineage_variants/digimon_autoloop`. The recovery manifest describes it as an autoloop-related lineage variant. The inventory records 94,670,392 readable bytes, 1,295 files, 121 directories, 33 symlinks, and git head `ERROR`. [1]

Focused ingest shows that this variant preserves a later operators-first DIGIMON state, not only an early autoloop experiment. It frames DIGIMON as composable GraphRAG for multi-hop QA: 28 typed operators, MCP/direct interfaces, two-model graph build/query architecture, and an explicit thesis that adaptive operator routing should beat fixed pipelines. See [Digimon Autoloop Operator Routing](/wiki/sources/digimon-autoloop-operator-routing.md) and [Adaptive Operator Routing](/wiki/concepts/adaptive-operator-routing.md).

# Shallow Contents

Shallow inspection shows Config/Core/Data/Doc/Option folders, acceptance gates, config, docs, eval, examples, hooks, investigations, prompts, specs, testing, tests, MCP tracker/plan/reference docs, and multiple test files.

# Verification Gap

The git head is recorded as `ERROR`; inspect non-destructively before making commit claims.

The `.git` file points to `/home/brian/projects/Digimon_for_KG_application/.git/worktrees/Digimon_for_KG_application__autoloop`, which is unavailable in this preserved location. That explains the failed git-head read and should be preserved as provenance damage rather than silently repaired. [2]

# Key Themes

- [Adaptive Operator Routing](/wiki/concepts/adaptive-operator-routing.md): later falsifiable thesis and negative development evidence.
- [MCP Autoloop Interface](/wiki/concepts/mcp-autoloop-interface.md): early MCP/UKRF plan lineage versus later MCP/direct supported interface.
- [Graph Build Manifest](/wiki/concepts/graph-build-manifest.md): graph attribute and tool-gating model for truthful capability exposure.
- [GraphRAG Upstream Lineage](/wiki/concepts/graphrag-upstream-lineage.md): explicit JayLZhou GraphRAG / DIGIMON foundation in README.

# Status Caveat

This variant contains both strong-looking 50-question subset benchmark claims and later negative controlled development evidence. The later plan and postmortem say the adaptive-routing thesis is not proven and should go through a constrained salvage pass before any locked evaluation claim. [3]

# Citations

[1] `../archive_full_record/metadata/recovery_inventory.tsv`
[2] `../archive_full_record/lineage_variants/digimon_autoloop/.git`  
[3] `../archive_full_record/lineage_variants/digimon_autoloop/docs/plans/03_prove_adaptive_routing.md`; `../archive_full_record/lineage_variants/digimon_autoloop/docs/reports/2026-03-19_musique_50q_postmortem.md`
