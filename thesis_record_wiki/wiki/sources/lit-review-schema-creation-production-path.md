---
type: Source
title: Lit Review Schema Creation Production Path
description: Source summary for schema_creation code and prompts that produced or evolved the lit-review theory extraction pipeline.
tags: [source, lit-review, schema-creation, prompts, production-path]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/schema_creation/prompts/README.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/schema_creation/fix_information_loss.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/schema_creation/prompt_loader.py
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/schema_creation/theory_extractor_v13_multi.py
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/schema_creation/theory_extractor_v13_single.py
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/schema_creation/multiphase_processor_improved.py
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/schema_creation/prompts/multiphase/phase1_vocabulary_extraction.txt
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/schema_creation/prompts/multiphase/phase2_ontological_classification.txt
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/schema_creation/prompts/multiphase/phase3_schema_generation.txt
confidence: medium
---

# Summary

`src/schema_creation` is the lit-review experiment's schema-extraction production-path area. This bounded slice samples the current/latest-looking v13 extractors, the improved multiphase processor, prompt externalization, the three multiphase prompts, and the explicit note about information loss between phases.

The directory contains 40 files and appears to preserve several generations of extractor experiments rather than one clean production module.

# Pipeline Shape

The multiphase prompt README defines two extraction modes:

- simple single-pass extraction for quick results
- multiphase extraction with Phase 1 vocabulary extraction, Phase 2 ontological classification, and Phase 3 schema generation [1]

The improved multiphase processor implements that three-phase path with Pydantic outputs, external prompt files, and model-type-specific YAML conversion. [6]

# Information-Loss Fix

`fix_information_loss.md` states the core problem clearly: Phase 1 might extract 61 terms, but Phase 3 was only seeing a Phase 2 summary and therefore lost terms. The fix is to pass the full Phase 1 vocabulary into Phase 3. [2]

`multiphase_processor_improved.py` appears to implement that fix by embedding `COMPLETE PHASE 1 VOCABULARY` into the Phase 3 classification summary and telling the schema prompt that all Phase 1 terms must be included. [6][9]

# v13 Extractor Line

The v13 extractor files introduce a separate full-paper path:

- `theory_extractor_v13_multi.py` identifies all theories in a paper, decides whether extraction should be single/nested/parallel, and extracts each theory with full paper text. [4]
- `theory_extractor_v13_single.py` explicitly focuses on the single primary theory and treats mechanisms/applications as parts of one unified theory. [5]

Both emphasize no truncation of PDF text. Both also use JSON object response format in places, so this is historically important code but should not be assumed to satisfy the later ecosystem-wide structured-output policy without review. [4][5]

# Prompt Design

The three multiphase prompts encode important design commitments:

- Phase 1 asks for exhaustive theoretical vocabulary, not a fixed number of terms. [7]
- Phase 2 classifies terms into entities, relationships, properties, actions, measures, modifiers, truth-values, and operators with specific domain/range constraints. [8]
- Phase 3 chooses among property graph, hypergraph, table/matrix, sequence, tree, timeline, or other, and explicitly says not to default to property graph. [9]

# Caveats

This source slice is production-path evidence, not runtime verification. It shows the intended and partially implemented pipeline behind generated artifacts like [Carter Theory Analysis Output](/wiki/sources/carter-theory-analysis-output.md), but it does not prove that a specific Carter output was produced by a specific code revision.

Archive-specific implementation caveats:

- many files contain hardcoded historical `/home/brian/...` paths
- some older files use `json_object` rather than strict schema response formats
- fallback imports and multiple extractor generations suggest experimental evolution rather than a stabilized package

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/schema_creation/prompts/README.md`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/schema_creation/fix_information_loss.md`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/schema_creation/prompt_loader.py`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/schema_creation/theory_extractor_v13_multi.py`  
[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/schema_creation/theory_extractor_v13_single.py`  
[6] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/schema_creation/multiphase_processor_improved.py`  
[7] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/schema_creation/prompts/multiphase/phase1_vocabulary_extraction.txt`  
[8] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/schema_creation/prompts/multiphase/phase2_ontological_classification.txt`  
[9] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/schema_creation/prompts/multiphase/phase3_schema_generation.txt`
