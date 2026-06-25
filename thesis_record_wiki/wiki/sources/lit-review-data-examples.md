---
type: Source
title: Lit Review Data And Examples
description: Inventory and interpretation of preserved lit_review data and examples folders, including paper texts, test texts, UI requirements, Carter examples, hybrid schemas, OWL/causal stress tests, and Grusch prompt material.
tags: [source, lit-review, data, examples, carter, semantic-hypergraph, grusch, owl, causal, hybrid]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/data/
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/examples/
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/examples/carter_young1996_faithful_analysis.yml
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/examples/example_social_ecological_systems_schema.yml
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/examples/owl_enhanced_stress_test_schema.yml
confidence: high
---

# Summary

The preserved `data/` directory contains 17 files with aggregate hash `e68dd68f8453de258f027af5eca7183310af721bfabf3dc330d702aa6e18bd2a`. The preserved `examples/` directory contains 11 files with aggregate hash `51a55d366f2b9be42dabc8d971627f7c91223eac2d15957535cb35d91d5d721b`. [1][2]

Together, these folders ground the lit-review experiment. They preserve source papers, benchmark text inputs, prompts, UI dependency notes, Carter cognitive-map examples, Social-Ecological Systems hybrid examples, OWL-enhanced stress tests, and causal-extension sketches.

# Data Folder

## Paper Texts

The `data/papers/` subtree includes:

- full Semantic Hypergraphs paper text
- truncated Semantic Hypergraphs text
- a synthetic simple hypergraph test paper [1]

These files ground the Semantic Hypergraph extraction and formal-notation work.

## Test Texts

The `data/test_texts/` subtree includes:

- Carter speech excerpts and full Carter speech files
- Ground News and Iran debate texts for Semantic Hypergraph/news-claim extraction
- Grusch testimony and an information-disorder prompt
- OpenAI structured-output documentation snapshot
- `problems_to_fix.txt`, which explicitly diagnoses property-graph anchoring and domain/range misuse
- `requirements_ui.txt`, which lists Streamlit/OpenAI/Pandas/Plotly/NetworkX/Numpy UI dependencies [1]

`problems_to_fix.txt` is especially important: it explains that the earlier prompt/template biased model selection toward property graphs and incorrectly encouraged domain/range fields on entity definitions. This is a direct source for the [Schema Extraction Pipeline Evolution](/wiki/concepts/schema-extraction-pipeline-evolution.md) and [Model Form Routing](/wiki/concepts/model-form-routing.md) pages.

# Examples Folder

## Carter / Young 1996 Examples

The Carter examples preserve multiple versions of the same target task:

- minimal output
- API analysis
- complete analysis
- proper analysis
- faithful Young 1996 analysis
- OWL reasoning and manual validation variants [2][3]

The faithful Young example contains 48 concepts and Young-style relationship categories, while the complete-analysis variant has different extracted concept counts and frequencies. This makes the folder useful for comparing fidelity targets, generated outputs, and schema drift.

## Hybrid And OWL Stress Tests

The examples also include:

- Social-Ecological Systems hybrid schema
- adaptive-governance hybrid stress-test schema
- OWL-enhanced stress-test schema
- causal-enhanced schema framework [4][5]

These examples show the thesis expanding beyond property graphs into hybrid, OWL, statistical, logical, and causal representations.

# Interpretation

This slice reinforces a key preservation rule: data and examples are not expendable clutter. They are the benchmark layer that explains what the code and docs were trying to produce.

The code directories show how extraction/application was implemented; these folders show what it was run on and what target outputs looked like.

# Caveats

- Some test files are copied external content or prompt snapshots, not original research outputs.
- `didnt_get.txt` appears to preserve a failed or inaccessible PDF-download URL/token trace; treat it as process residue.
- The UI requirements file lives under `data/test_texts/`, not next to the UI launcher that references `requirements_ui.txt`.
- Examples are not all mutually consistent; that inconsistency is part of the record of schema variant drift.

# Links

- [Lit Review Docs Bundle](/wiki/sources/lit-review-docs-bundle.md)
- [Lit Review Visualization Code](/wiki/sources/lit-review-visualization-code.md)
- [Lit Review Semantic Hypergraph Schema Family](/wiki/sources/lit-review-semantic-hypergraph-schema-family.md)
- [Lit Review Young1996 Schema Family](/wiki/sources/lit-review-young1996-schema-family.md)
- [Schema Variant Drift](/wiki/concepts/schema-variant-drift.md)
- [Model Form Routing](/wiki/concepts/model-form-routing.md)
- [Formal Notation As Theory Content](/wiki/concepts/formal-notation-as-theory-content.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/data/`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/examples/`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/examples/carter_young1996_faithful_analysis.yml`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/examples/example_social_ecological_systems_schema.yml`  
[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/examples/owl_enhanced_stress_test_schema.yml`
