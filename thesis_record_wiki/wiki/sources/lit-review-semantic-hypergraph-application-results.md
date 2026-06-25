---
type: Source
title: Lit Review Semantic Hypergraph Application Results
description: Semantic-hypergraph extraction/application results, critiques, visualizations, and scripts from the lit-review experiment.
tags: [source, lit-review, semantic-hypergraph, extraction, application, theory]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/results/semantic_hypergraph/approach_comparison.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/results/semantic_hypergraph/option2_critique.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/results/semantic_hypergraph/multi_pass_vs_option2_critique.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/results/semantic_hypergraph/fixed_schema_critique.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/results/semantic_hypergraph/enhanced_extraction/comparison_report.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/results/semantic_hypergraph/enhanced_extraction/sh_notation_simple.txt
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/results/multi_pass_extraction_summary.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/results/enhanced_extraction_final_report.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/schema_application/apply_semantic_hypergraph_to_iran.py
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/run_semantic_hypergraph_extraction.py
confidence: high
---

# Summary

This slice covers the semantic-hypergraph branch of the lit-review experiment: extracting the Semantic Hypergraph paper into a usable schema, applying it to texts, and diagnosing extraction fidelity failures.

The core lesson is methodological: for formal theories like Semantic Hypergraph, notation systems, type codes, role codes, pattern syntax, and parsing algorithms are not optional implementation details. They are part of the theory. Treating them as secondary caused major fidelity loss.

# Extraction Approaches

The approach comparison contrasts two extraction strategies. Option 1 placed notation into a unified structured schema and was API-friendly, but simplified role codes, missed role details, and reduced pattern syntax. Option 2 separated implementation extraction from theory schema and captured more role codes, exact definitions, pattern variables, and complete examples. [1]

The recommendation was hybrid: use integrated schema extraction for structured theory content, then run a separate implementation/notation extraction and merge the notation details back into the schema. [1]

# Critique Chain

The Option 2 critique shows that even the better separate extraction still missed critical material from the original paper:

- atom vs non-atom distinctions in the type system
- formal type inference rules such as `(M x) -> x` and `(P [C R S]+) -> R`
- predicate and builder role systems
- wildcard/pattern syntax
- alpha-stage and beta-stage algorithm details
- evaluation metrics and application-specific patterns [2]

The fixed-schema critique says the fixed multiphase processor solved one problem, preserving 67 extracted terms rather than losing terms during schema construction, but still did not extract the notation system and pattern language. Its root-cause diagnosis is that the model treated schema as entities/relationships/properties while the paper treated schema as a complete formal system. [4]

The multi-pass critique shows another failure mode. The multi-pass system improved algorithms, metrics, examples, and symbols, but completely missed the key plain-text tables containing type codes, inference rules, and argument roles. It reported 0 type codes, 0 role codes, and 0 tables for the Semantic Hypergraph paper, even though those were central to implementation. [3][7]

# Enhanced Result

The enhanced extraction comparison reports the later improvement:

- captures 13 argument-role codes
- includes special symbols such as `+/B` and `:/J`
- documents pattern syntax
- includes eight type-inference rules
- adds implementation algorithms for hyperedge creation and pattern matching [5]

The accompanying notation inventory lists the type system, role suffixes, wildcards, quantifiers, delimiters, and figure glyphs. [6]

# Application Artifacts

The preserved application script for Iran debate analysis builds a prompt from classification, telos, process, ontology, and axioms, then asks for alpha classification, beta transformation, and decomposed relations. It hardcodes `/home/brian/lit_review/...` paths and uses `OpenAI()` directly, so it is historical application evidence rather than current runnable infrastructure. [9]

The standalone extraction runner similarly adds local `src` to `sys.path`, runs a three-phase schema extraction, and writes debug outputs and YAML under local paths. This preserves the extraction workflow shape but should not be assumed runnable without reconstructing that historical environment. [10]

Visualized instances show concrete outputs:

- Ground News instance has atoms such as `spain_pm_sanchez/C`, `gaza/C`, and predicates like `describe/P`, `condemn/P`, `kill/P`, plus a main conjunction hyperedge. [11]
- Iran debate enhanced instance includes typed role notation such as `applaud/P.sa`, `warn/P.soa`, `invite/P.so`, and hyperedges tagged with claims like `<claim_Bret>` and `<claim_Rosemary>`. [12]

# Relation To Prior Lit-Review Pages

This slice extends [Lit Review Schema Creation Production Path](/wiki/sources/lit-review-schema-creation-production-path.md) by showing the downstream failure cases that motivated the "operational knowledge" and notation-preservation improvements.

It also extends [Schema Extraction Pipeline Evolution](/wiki/concepts/schema-extraction-pipeline-evolution.md) with a concrete rule: for formal theories, extraction must include implementation notation and algorithms as theory content.

# Links

- [Formal Notation As Theory Content](/wiki/concepts/formal-notation-as-theory-content.md)
- [Schema Extraction Pipeline Evolution](/wiki/concepts/schema-extraction-pipeline-evolution.md)
- [Automated Theory Extraction](/wiki/concepts/automated-theory-extraction.md)
- [Lit Review Schema Creation Production Path](/wiki/sources/lit-review-schema-creation-production-path.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/results/semantic_hypergraph/approach_comparison.md`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/results/semantic_hypergraph/option2_critique.md`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/results/semantic_hypergraph/multi_pass_vs_option2_critique.md`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/results/semantic_hypergraph/fixed_schema_critique.md`  
[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/results/semantic_hypergraph/enhanced_extraction/comparison_report.md`  
[6] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/results/semantic_hypergraph/enhanced_extraction/sh_notation_simple.txt`  
[7] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/results/multi_pass_extraction_summary.md`  
[8] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/results/enhanced_extraction_final_report.md`  
[9] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/schema_application/apply_semantic_hypergraph_to_iran.py`  
[10] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/run_semantic_hypergraph_extraction.py`  
[11] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/results/semantic_hypergraph/ground_news_sh_visualization.txt`  
[12] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/results/semantic_hypergraph/iran_debate/iran_debate_sh_visualization.txt`
