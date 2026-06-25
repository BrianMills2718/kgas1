---
type: LineageVariant
title: Digimons Docs
description: Documentation extraction variant.
tags: [variant, digimons, docs]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/metadata/recovery_inventory.tsv
confidence: high
---

# Summary

`Digimons_docs` is preserved at `../archive_full_record/lineage_variants/Digimons_docs`. The recovery manifest describes it as a documentation extraction variant. The inventory records 10,939,697 readable bytes, 928 files, 310 directories, and git head `269088673707f7b0f5b9138e7d5bcd8e1117b0e9`. [1]

# Interpretation

This may be a high-leverage source for reconstructing the conceptual/documentation state without scanning the larger code-heavy archives first.

Because it is small and documentation-focused, this should probably be ingested before the larger `digimons_old` and `digimon_lineage_Digimons` trees. Its pages can establish a vocabulary for later variant comparison.

Initial ingest shows this variant is especially important for [Documentation Status Truthfulness](/wiki/concepts/documentation-status-truthfulness.md): it explicitly separates target architecture from implementation status and assigns current status to roadmap docs. Its conservative roadmap records a verified 2025-07-31 baseline rather than treating all architectural goals as implemented. [2]

# Related Pages

- [Research Lineage](/wiki/concepts/research-lineage.md)
- [Recovery Inventory](/wiki/sources/recovery-inventory.md)
- [Digimons Documentation Repository](/wiki/sources/digimons-docs-documentation-repository.md)

# Citations

[1] `../archive_full_record/metadata/recovery_inventory.tsv`  
[2] `../archive_full_record/lineage_variants/Digimons_docs/docs/roadmap/ROADMAP_OVERVIEW_CONSERVATIVE.md`
