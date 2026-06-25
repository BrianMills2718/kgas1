---
type: LineageVariant
title: Digimon Core Sparse
description: Sparse / contract-oriented variant with many evidence and architecture documents.
tags: [variant, digimon, contracts, evidence]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/metadata/recovery_inventory.tsv
confidence: medium
---

# Summary

`digimon_core_sparse` is preserved at `../archive_full_record/lineage_variants/digimon_core_sparse`. The recovery manifest describes it as a sparse / contract-oriented variant. The inventory records 24,025,718 readable bytes, 1,116 files, 118 directories, and git head `ERROR`. [1]

This variant is now one of the most important small slices because it preserves the contract-first migration record: competing tool interfaces, structured contracts, evidence files, and stress-test bottlenecks. It should be read with [Digimon Core Sparse Contract Layer](/wiki/sources/digimon-core-sparse-contract-layer.md).

# Shallow Contents

Shallow inspection shows contracts, config, docs, src, and numerous evidence / architecture markdown files such as `ARCHITECTURAL_BOTTLENECKS_ANALYSIS.md`, `Contract_First_Action_Plan.md`, and many `Evidence_*` files.

# Verification Gap

The git head is recorded as `ERROR`; inspect non-destructively before making branch or commit claims.

Despite the git-head gap, the file names suggest this variant may be important for reconstructing contract-first thinking, architectural bottlenecks, and evidence-driven implementation claims. It should feed [Research Lineage](/wiki/concepts/research-lineage.md) after a focused evidence-doc ingest.

The `.git` file points to `/home/brian/projects/Digimons/.git/worktrees/digimon_core_sparse`, which is not available in this preserved location. That explains the inventory's `ERROR` git head and should be preserved as provenance damage, not silently repaired.

# Key Themes

- [Contract-First Migration](/wiki/concepts/contract-first-migration.md)
- [Relationship Extraction Bottleneck](/wiki/concepts/relationship-extraction-bottleneck.md)
- [GraphRAG Upstream Lineage](/wiki/concepts/graphrag-upstream-lineage.md)
- Evidence-based implementation tracking with `Evidence_*.md` files.

# Citations

[1] `../archive_full_record/metadata/recovery_inventory.tsv`  
[2] `../archive_full_record/lineage_variants/digimon_core_sparse/.git`  
[3] `../archive_full_record/lineage_variants/digimon_core_sparse/Contract_First_Uncertainty_Analysis.md`  
[4] `../archive_full_record/lineage_variants/digimon_core_sparse/ARCHITECTURAL_BOTTLENECKS_ANALYSIS.md`
