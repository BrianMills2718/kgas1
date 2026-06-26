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

This can become a focused regression test later, but this page intentionally does not add test code. It records the validated minimum slice and its boundaries first.

# Stop Lines

Do not expand this smoke into any of the following without a separate plan:

- Neo4j-backed pipeline execution;
- archived Twitter-like dataset reads;
- provider/LLM calls;
- broad PageRank or ontology-aware extractor execution;
- mutation of raw archive material;
- claims about ADR-029/Comprehensive7 current implementation.

# Recommended Next Test Slice

If implementation verification becomes the priority, the safest next code change would be a focused no-IO unit test for the confidence package:

- create CERQual confidence;
- set a range;
- combine with medium confidence;
- assert returned values are bounded;
- assert metadata records the expected method/source;
- assert no database or provider dependencies are imported.

That would upgrade the evidence from observed smoke to repeatable test, while staying independent of Neo4j and raw thesis archives.

# Relationship To Wiki

- [Current Uncertainty Code Path Map 2026 06 26](current-uncertainty-code-path-map-2026-06-26.md): source-level map that selected this smoke as the smallest safe current-code slice.
- [Uncertainty Framework Consolidation 2026 06 26](/wiki/concepts/uncertainty-framework-consolidation-2026-06-26.md): high-level uncertainty lineage map.
- [Current Status Verification Discipline](/wiki/concepts/current-status-verification-discipline.md): why this page labels the observed result as a smoke, not full runtime proof.
- [Runtime Verification Isolation Boundary](/wiki/concepts/runtime-verification-isolation-boundary.md): relevant for any future Neo4j-backed expansion.

# Citations

[1] `../src/core/confidence_score.py`  
[2] `../src/core/confidence_scoring/`  
[3] `/wiki/sources/current-uncertainty-code-path-map-2026-06-26.md`
