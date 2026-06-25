---
type: Source
title: Digimon Lineage Config Schemas
description: Inventory and interpretation of config/schemas theory meta-schema versions, concrete theory schemas, and tool contract schema.
tags: [source, digimon-lineage, config, schemas, theory-meta-schema, theory-application, tool-contracts]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/config/schemas/
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/config/schemas/THEORY_META_SCHEMA_V11_CHANGES.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/config/schemas/theory_meta_schema_v12.json
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/config/schemas/tool_contract_schema.yaml
confidence: high
---

# Summary

`config/schemas/` is a 27-file, 396,153-byte schema bundle with aggregate hash `7138bcac7708f9d2415a805af7eec5cd8d529a707f49839fcabbc589dc38663b`. [1]

It contains:

- five theory meta-schema versions in the active folder: v9, v10, v11, v11.1, and v12 [1]
- archived copies of the same meta-schema versions, plus `docs_theory_meta_schema_v9/10` variants [1]
- `THEORY_META_SCHEMA_V11_CHANGES.md`, a rationale document for the v11 shift toward theoretical honesty and applicability checks [2]
- twelve concrete theory-schema examples spanning agenda setting, balance theory, cognitive dissonance, framing, institutional theory, network theory, prospect theory, rational choice, self-categorization, ELM, uses and gratifications, and Young/WorldView cognitive mapping [1]
- `tool_contract_schema.yaml`, the validation schema for tool/adapter contracts [4]

# Meta-Schema Versions

| File | Title / description | Notes |
|---|---|---|
| `theory_meta_schema_v9.json` | computable social-science theory framework with DOLCE alignment | earlier ontology/process/telos form |
| `theory_meta_schema_v10.json` | executable social-science theory framework with practical implementation details | adds execution/configuration emphasis |
| `theory_meta_schema_v11.json` | enhanced framework with explicit scope and applicability requirements | adds theoretical scope, empirical foundation, theory comparisons, applicability checks |
| `theory_meta_schema_v11_1.json` | refined two-stage analysis framework with practical usability improvements | preserves scope/foundation but simplifies the v11 direction |
| `theory_meta_schema_v12.json` | value-driven two-layer architecture for theoretical structure and computational analysis | shifts toward preserving author terminology, logical structure, computational representation, algorithms, extraction process, validation, and metadata tracking [3] |

The archived duplicates preserve version history rather than separate substantive content. [1]

# V11 Theoretical Honesty Shift

`THEORY_META_SCHEMA_V11_CHANGES.md` says v11 was introduced after Social Identity Theory analysis exposed a core failure mode: applying a competitive intergroup theory to diplomatic cooperation without systematically checking scope. [2]

The v11 changes add:

- required `theoretical_scope`
- `empirical_foundation`
- `theory_comparisons`
- mandatory execution-time applicability check
- negative-finding interpretation
- `post_hoc_enhancements`
- source-verification metadata [2]

The document states the core conceptual change as moving from "how to apply any theory" toward deciding whether and how a theory should be applied. This connects directly to [Model Form Routing](/wiki/concepts/model-form-routing.md), [Prompt Calibration Loop](/wiki/concepts/prompt-calibration-loop.md), and [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md): schema validity is not enough if the theory is inappropriate for the case.

# Concrete Theory Schemas

The active concrete theory schemas preserve examples across multiple social-science traditions:

| File | Theory name |
|---|---|
| `agenda_setting_theory_schema.json` | Agenda Setting Theory |
| `balance_theory_heider_schema.json` | Heider's Balance Theory |
| `cognitive_dissonance_theory_schema.json` | Cognitive Dissonance Theory |
| `framing_theory_schema.json` | Framing Theory |
| `institutional_theory_schema.json` | Institutional Theory |
| `network_theory_schema.json` | Network Theory / Social Network Analysis |
| `prospect_theory_schema.json` | Prospect Theory |
| `rational_choice_theory_schema.json` | Rational Choice Theory |
| `self_categorization_theory_schema.json` | Self-Categorization Theory |
| `test_elm_theory.json` | Elaboration Likelihood Model |
| `uses_gratifications_theory_schema.json` | Uses and Gratifications Theory |
| `young_cognitive_mapping_schema.json` / `young_cognitive_mapping_v11_1.json` | Young/WorldView cognitive mapping with semantic-network formalism |

Most theory examples use a shared top-level shape: `theory_id`, `theory_name`, version metadata, `telos`, `classification`, `ontology`, `execution`, `validation`, and `mode`. The v11.1-style examples add `theoretical_scope` and `empirical_foundation`. [1]

# Tool Contract Schema

`tool_contract_schema.yaml` defines the structure that tool and adapter contracts must follow, including input and output contracts. It is the config-side sibling of the concrete tool contracts summarized in [Digimon Lineage Config And Contracts](/wiki/sources/digimon-lineage-config-contracts.md). [4]

# Interpretation

This config schema bundle is separate from the lit-review schema corpus:

- the lit-review schema corpus preserves generated/extracted theory artifacts from papers
- `config/schemas/` preserves framework-level meta-schema versions and curated/example schemas intended to drive KGAS configuration and validation

The most important thesis lesson is the v11 applicability shift. The project was no longer only asking "can we encode theory as JSON"; it was asking "should this theory be applied to this empirical context, and how do we preserve negative findings without forcing a fit?"

# Links

- [Digimon Lineage Config And Contracts](/wiki/sources/digimon-lineage-config-contracts.md)
- [Lit Review Schemas Corpus Inventory](/wiki/sources/lit-review-schemas-corpus-inventory.md)
- [Model Form Routing](/wiki/concepts/model-form-routing.md)
- [Prompt Calibration Loop](/wiki/concepts/prompt-calibration-loop.md)
- [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md)
- [Schema Extraction Pipeline Evolution](/wiki/concepts/schema-extraction-pipeline-evolution.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/config/schemas/`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/config/schemas/THEORY_META_SCHEMA_V11_CHANGES.md`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/config/schemas/theory_meta_schema_v12.json`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/config/schemas/tool_contract_schema.yaml`
