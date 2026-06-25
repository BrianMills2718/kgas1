---
type: Source
title: Digimon Lineage Ops Scaffolding
description: Inventory and interpretation of preserved GitHub Actions, Docker deployment files, and requirements lock sets in the large Digimons lineage.
tags: [source, digimon-lineage, ops, github-actions, docker, requirements, ci, deployment]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/.github/
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/docker/
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/requirements/
confidence: medium
---

# Summary

The large `digimon_lineage_Digimons` variant preserves a concrete operations scaffold:

- `.github/`: 12 files, 65,100 bytes, aggregate hash `f764146483f04e975bdb0816c735e98f8bca21370d787b7b36a84af9ee90f982` [1]
- `docker/`: 6 files, 10,504 bytes, aggregate hash `3543e59f25a82c29abc4069d75a9c6c923bc1181cde4e792a3ef9de4d3cce382` [2]
- `requirements/`: 9 files, 41,392 bytes, aggregate hash `a1d1c84cb98f3fb24339530d595dfdc90644aee4cfab38517885c856b17262a0` [3]

This is evidence that the lineage attempted CI, documentation governance, interface validation, production deployment, containerization, and dependency locking. It is not by itself evidence that those workflows passed in GitHub or that the deployment was production-safe.

# GitHub Actions

The `.github/workflows/` directory preserves six workflow files:

| Workflow | Name | Triggers | Jobs |
|---|---|---|---|
| `ci-tests.yml` | CI Tests | push, pull request | `test` |
| `docs-ci.yml` | Docs Quality Checks | pull request | `docs_checks` |
| `docs_check.yml` | Documentation Governance | pull request, push | `docs_lint`, `archive_guard` |
| `integration.yml` | Integration Tests | push, pull request | `verification-matrix`, `integration-full`, `doc-governance` |
| `interface-validation.yml` | Interface Validation | push, pull request | `validate-interfaces` |
| `production-deploy.yml` | Production Deployment | push, pull request | `test`, `build`, `deploy` |

The preserved PR templates emphasize docs placement rules, docs checks, testing notes, roadmap/status updates, and passing CI. [1]

This connects to the later [Current Status Verification Discipline](/wiki/concepts/current-status-verification-discipline.md): the repository was adding process gates around docs, interfaces, and integration, but a workflow file is only intended enforcement until run evidence is tied to it.

# Docker Deployment Scaffold

The Docker slice includes:

- root `Dockerfile`
- `docker-compose.production.yml`
- `docker/production/Dockerfile.prod`
- `docker/production/start-production.sh`
- local instructions in `docker/CLAUDE.md` / `docker/AGENTS.md` [2]

The compose file wires API, Neo4j, Redis, Prometheus, and Grafana services. The production startup script waits for Neo4j, runs `ProductionValidator`, initializes Neo4j schema, starts `src/mcp_server.py`, then starts Streamlit. [2]

Caveat: the preserved compose/startup files include hardcoded default deployment credentials/placeholders and assume service names, scripts, and entry points that require current-runtime verification before use. They should be treated as historical deployment scaffolding, not a safe production recipe.

# Requirements Layout

The requirements directory centralizes `pip-tools` inputs and lock files:

| File group | Purpose |
|---|---|
| `base.in` / `base.txt` | production/core dependencies: MCP, Neo4j, SQLAlchemy, FAISS, Redis, NLP/ML, Streamlit, monitoring, encryption |
| `dev.in` / `dev.txt` | base plus pytest, coverage, formatting/type/lint tools, pip-tools |
| `llm.in` / `llm.txt` | base plus Gemini, OpenAI, tokenization, retry, and LangChain layers |
| `ui.in` / `ui.txt` | base plus Streamlit UI, Plotly/Pandas, file/PDF helpers, RDF tooling |
| `README.md` | explains edit `.in`, run `pip-compile`, commit `.in` and `.txt`, install selected lock file [3] |

Caveat: the generated `*.txt` files preserve absolute historical path comments under Brian's local `Digimons` path. Those comments are provenance artifacts from `pip-compile`, not portable installation instructions. [3]

# Interpretation

This slice shows the system trying to move from local research code toward agent- and CI-governed engineering:

- docs governance moved into PR templates and workflows
- interface validation became a CI surface
- integration and production deployment were expressed as workflows
- deployment assumptions were encoded into Docker/Compose
- dependencies were split into core/dev/LLM/UI lock sets

The main risk is status inflation. These files are strong evidence of intended operational discipline, but they must be paired with CI run logs, runtime import checks, and deployment evidence before being cited as proof that KGAS was deployable or stable.

# Links

- [Digimon Lineage Active State](/wiki/sources/digimon-lineage-active-state.md)
- [Current Runtime Import Check 2026-06-25](/wiki/sources/current-runtime-import-check-2026-06-25.md)
- [Current Runtime Repair Plan 2026-06-25](/wiki/sources/current-runtime-repair-plan-2026-06-25.md)
- [Current Status Verification Discipline](/wiki/concepts/current-status-verification-discipline.md)
- [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md)
- [Academic Proof Of Concept Scope](/wiki/concepts/academic-proof-of-concept-scope.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/.github/`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/docker/`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/requirements/`
