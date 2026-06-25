---
type: LineageVariant
title: Digimon Lineage Digimons
description: Largest preserved lineage bundle and likely central source for later Digimons evolution.
tags: [variant, digimons, lineage, large-archive]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/metadata/recovery_inventory.tsv
confidence: medium
---

# Summary

`digimon_lineage_Digimons` is preserved at `../archive_full_record/lineage_variants/digimon_lineage_Digimons`. The recovery manifest identifies it as the largest preserved lineage bundle. The inventory records 9,334,155,151 readable bytes, 78,514 files, 9,619 directories, 157 symlinks, and git head `e14164e818588b474b68eb0e525ea78d9a10ce1c`. [1]

Focused root ingest shows this bundle preserves a September 2025 active KGAS state with documentation truthfulness, roadmap consolidation, and vertical-slice/main-system reconciliation work. The first slice should be read through [Digimon Lineage Active State](/wiki/sources/digimon-lineage-active-state.md), [Reality Verification Arc](/wiki/concepts/reality-verification-arc.md), and [Vertical Slice vs Main System](/wiki/concepts/vertical-slice-vs-main-system.md).

The `tool_compatability` slice shows the same type-based composition decision seen in `Digimons_clean_for_real`, but with active local context: the subtree is explicitly marked as the adapter/framework direction and the nested `poc/vertical_slice` path is marked as the active proof-of-concept framework. See [Digimon Lineage Tool Compatibility](/wiki/sources/digimon-lineage-tool-compatibility.md).

The architecture-docs slice confirms that `docs/architecture/` is target design, not status tracking, and adds a critical internal review of uncertainty/provenance over-engineering. See [Digimon Lineage Architecture Docs](/wiki/sources/digimon-lineage-architecture-docs.md) and [Uncertainty Traceability Architecture](/wiki/concepts/uncertainty-traceability-architecture.md).

# Shallow Contents

Shallow inspection shows archive, archived, config, contracts, data, dev, docker, docs, evidence, examples, experiments, investigation, logs, requirements, research, scripts, src, tests, tool compatibility material, tools, and recovered UI components.

# Interpretation

This is likely the highest-value lineage source after the current repo. Continue ingesting in slices: root README/CLAUDE/roadmap, tool compatibility, architecture docs, evidence archives, experiments, and recovered UI.

# Current Git Caveat

At focused ingest time, the preserved git repo reported `master...kgas/master [ahead 1]` with a modified nested `experiments/tool_compatability/GraphRAG` path. The recovery inventory's git head is therefore only one preserved reference point; later local working-tree state should be investigated before making exact commit-history claims. [2]

# Citations

[1] `../archive_full_record/metadata/recovery_inventory.tsv`
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/.git` and `git status` output observed during 2026-06-25 ingest.
