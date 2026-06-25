---
type: SourceSummary
title: Digimon Lineage Doc Generation Scripts
description: Small archive of architecture-document generation scripts for concatenating KGAS architecture docs, splitting generated bundles, extracting ADRs, and requesting Gemini architecture review.
tags: [source, digimon-lineage, archive, docs, generation, architecture, adr, gemini-review]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/doc_generation_scripts/
confidence: high
---

# Summary

`archive/doc_generation_scripts/` is a four-file, 26,430-byte archive of KGAS architecture-document generation scripts. Its aggregate content-manifest hash is `16596e7b7ffd425e03c53b85779932f6d1c77cba677ecf374477834b993d4e84`. [1]

The scripts explain part of the generated-doc provenance: architecture docs were concatenated into comprehensive bundles, split back into themed documents, ADRs were extracted/indexed, and Gemini was asked to review selected architecture documentation. [2] [3] [4] [5]

# Inventory

| File | Bytes | Role |
| --- | ---: | --- |
| `architecture_review.py` | 4,154 | Collects selected architecture docs and sends them to Gemini for critique. [2] |
| `concatenate_architecture_docs.py` | 4,161 | Concatenates KGAS architecture docs into `KGAS_COMPREHENSIVE_ARCHITECTURE.md`. [3] |
| `split_architecture_docs.py` | 9,473 | Splits the comprehensive architecture doc into four logical section files and an index. [4] |
| `extract_all_adrs.py` | 8,642 | Extracts all ADR markdown files into a comprehensive ADR document and quick-reference index. [5] |

# Generated Architecture Bundles

`concatenate_architecture_docs.py` defines a prioritized reading order for architecture docs: architecture overview, glossary, project structure, CLAUDE notes, theoretical foundation, cross-modal analysis, agent interface, component architecture, bi-store justification, specifications, selected ADRs, concepts, schemas, MCP integration, and limitations. It writes a single target-architecture reference document. [3]

`split_architecture_docs.py` takes that comprehensive architecture document and splits it into four sections: core architecture and vision, theoretical framework and data architecture, analysis/integration architecture, and architecture decisions/extensions. [4]

# ADR Extraction

`extract_all_adrs.py` scans `docs/architecture/adrs` for `ADR-*.md`, sorts by ADR number, generates a complete ADR document, and creates a categorized quick-reference index with decision areas such as tool/interface architecture, data/storage architecture, analysis/processing architecture, integration/protocol architecture, and research/quality architecture. [5]

# Gemini Review

`architecture_review.py` collects a smaller set of key architecture docs and asks Gemini for a critical review focused on technical depth, implementation feasibility, architectural consistency, documentation quality, missing critical elements, and academic research suitability. It expects a `GEMINI_API_KEY` in the environment and writes `architecture_review_results.md`. [2]

# Caveats

These scripts are path-sensitive and assume they are run from a KGAS checkout with the expected `docs/architecture` layout. They generate or overwrite local markdown outputs in the working directory. [3] [4] [5]

The targeted credential scan found no literal OpenAI or Google API keys in this archive. [1]

# Interpretation

This archive is useful provenance for why later KGAS archives contain large generated architecture bundles and generated review artifacts. It also shows a documentation workflow risk: generated comprehensive docs can multiply architecture claims unless the generated files are clearly labeled as target architecture or review inputs rather than current implementation truth.

# Relationship To Wiki

- [Digimon Lineage Docs Architecture Cleanup 2025 08 29](digimon-lineage-docs-architecture-cleanup-2025-08-29.md): later cleanup archive for generated duplicate docs and architecture-doc clutter.
- [Digimon Lineage Gemini Review Tool Archive](digimon-lineage-gemini-review-tool-archive.md): broader validation/review tooling archive.
- [Digimon Lineage Architecture Docs](digimon-lineage-architecture-docs.md): large-lineage architecture docs overview.
- [Digimon Lineage Architecture ADRs Map](digimon-lineage-architecture-adrs-map.md): ADR decision-history slice.
- [Documentation Status Truthfulness](../concepts/documentation-status-truthfulness.md): recurring problem of separating target architecture from verified status.

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/doc_generation_scripts/`

[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/doc_generation_scripts/architecture_review.py`

[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/doc_generation_scripts/concatenate_architecture_docs.py`

[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/doc_generation_scripts/split_architecture_docs.py`

[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/doc_generation_scripts/extract_all_adrs.py`
