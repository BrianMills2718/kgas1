---
type: SourceSummary
title: Current Confidence Smoke Rerun Plan 2026 06 26
description: Non-destructive rerun plan and observed no-IO smoke result for the current confidence-scoring package.
tags: [source, verification, current-code, confidence, rerun-plan]
created: 2026-06-26
updated: 2026-06-26
sources:
  - ../src/core/confidence_score.py
  - ../src/core/confidence_scoring/
  - /wiki/sources/current-uncertainty-code-path-map-2026-06-26.md
confidence: high
---

# Summary

This page records the smallest safe current-code uncertainty rerun identified after the current code-path map: a no-IO smoke of the confidence-scoring package. It does not touch Neo4j, raw archives, provider APIs, or current source files. [1][2][3]

The smoke was run on 2026-06-26 in the project `.venv` and imported:

- `ConfidenceScore`
- `get_confidence_scoring_info`
- `ConfidenceFactory`
- `ConfidenceCalculator`

It then created a CERQual-backed score, set a confidence range, combined it with a medium-confidence score, and printed a small result dictionary. [1][2]

# Observed Smoke Result

Command shape:

```bash
. .venv/bin/activate
python - <<'PY'
from src.core.confidence_score import ConfidenceScore, get_confidence_scoring_info
from src.core.confidence_scoring import ConfidenceFactory, ConfidenceCalculator

score = ConfidenceScore.create_with_cerqual(
    methodological_limitations=0.8,
    relevance=0.7,
    coherence=0.75,
    adequacy_of_data=0.65,
    evidence_weight=2,
    source="smoke_test",
)
range_score = score.set_confidence_range(0.55, 0.85)
combined = range_score.combine_with(ConfidenceScore.create_medium_confidence())
print({
    "module": get_confidence_scoring_info()["module"],
    "score": round(score.value, 3),
    "range": range_score.confidence_range,
    "combined": round(combined.value, 3),
    "method": combined.propagation_method.value,
})
PY
```

Observed output:

```text
{'module': 'confidence_score', 'score': 0.725, 'range': (0.55, 0.85), 'combined': 0.974, 'method': 'bayesian_evidence_power'}
```

This proves only that the current package imports and the selected no-IO operations execute in the current environment. It does not prove pipeline-wide uncertainty propagation, ADR-029, external validation, or dissertation-level measurement validity.

# Minimal Rerun Scope

The safe rerun scope is:

1. import the confidence package;
2. instantiate `ConfidenceScore` through a public factory;
3. exercise CERQual construction;
4. exercise range assignment;
5. exercise one combination method;
6. assert the method is `bayesian_evidence_power` and values remain in `[0, 1]`.

This minimum slice is now repeatable as `../tests/current_runtime/test_confidence_scoring_smoke.py`. [4]

Verified test command:

```bash
. .venv/bin/activate && PYTHONPATH=. pytest -q tests/current_runtime/test_confidence_scoring_smoke.py
```

Observed test result: `1 passed` with 13 Pydantic v2 deprecation warnings from the existing confidence model and wrapper methods. The warnings concern v1-style validators, class-based config, `json_encoders`, and `.dict()` calls. They do not fail the smoke test, but they are a future maintenance caveat for the confidence package.

# Stop Lines

Do not expand this smoke into any of the following without a separate plan:

- Neo4j-backed pipeline execution;
- archived Twitter-like dataset reads;
- provider/LLM calls;
- broad PageRank or ontology-aware extractor execution;
- mutation of raw archive material;
- claims about ADR-029/Comprehensive7 current implementation.

# Recommended Next Test Slice

The focused no-IO unit test now covers:

- create CERQual confidence;
- set a range;
- combine with medium confidence;
- assert returned values are bounded;
- assert metadata records the expected method.

This upgrades the evidence from observed smoke to repeatable test, while staying independent of Neo4j and raw thesis archives.

# Relationship To Wiki

- [Current Uncertainty Code Path Map 2026 06 26](current-uncertainty-code-path-map-2026-06-26.md): source-level map that selected this smoke as the smallest safe current-code slice.
- [Uncertainty Framework Consolidation 2026 06 26](/wiki/concepts/uncertainty-framework-consolidation-2026-06-26.md): high-level uncertainty lineage map.
- [Current Status Verification Discipline](/wiki/concepts/current-status-verification-discipline.md): why this page labels the observed result as a smoke, not full runtime proof.
- [Runtime Verification Isolation Boundary](/wiki/concepts/runtime-verification-isolation-boundary.md): relevant for any future Neo4j-backed expansion.

# Citations

[1] `../src/core/confidence_score.py`  
[2] `../src/core/confidence_scoring/`  
[3] `/wiki/sources/current-uncertainty-code-path-map-2026-06-26.md`
[4] `../tests/current_runtime/test_confidence_scoring_smoke.py`
