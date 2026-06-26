# Analyze Endpoint Document Pipeline Investigation

Date: 2026-06-25

## Question

What real document extraction path should eventually back `/api/analyze`, now that metadata-only placeholder analysis has been disabled?

## Atoms

| ID | Question | Dependencies | Status |
|---|---|---|---|
| A1 | Does `/api/analyze` currently call a real document parser? | none | Answered |
| A2 | Which current code path performs real document loading plus downstream KG extraction? | none | Answered |
| A3 | Can that path be safely wired to `/api/analyze` as a small follow-up? | A1, A2 | Answered |

## Evidence

### A1: Current API behavior

`src/api/cross_modal_api.py` now validates upload extension and enum parameters, then returns explicit 501 with a message that `/api/analyze` is not wired to the current document extraction pipeline. This replaced the previous temp-file plus metadata-only placeholder graph path. [1]

Focused tests assert that valid-looking analyze requests return 501 without loading the registry, unsupported file types return 400, and invalid target formats return 400. [2]

Answer: `/api/analyze` does not currently call a parser, by design. That is the honest status until a real extraction adapter is built.

### A2: Candidate real document path

`CompleteGraphRAGPipeline.execute_complete_pipeline(document_path, ...)` is the strongest current candidate. It sequences document loading, chunking, entity extraction, relationship extraction, graph build, PageRank/query work, and returns structured stage/output data. [3]

The document-loading step uses `T01PDFLoaderUnified` through `_execute_document_loading(...)`, passing a `ToolRequest` with `file_path`. It expects tool output keys such as `text_content`, `document_ref`, confidence, page count, and processing method. [4]

`T01PDFLoaderUnified` performs real extraction for `.pdf` via `pypdf` and `.txt` via text loading; other extensions return `INVALID_FILE_TYPE`. Its contract description says "PDF or text file", despite the API previously accepting `.docx`, `.doc`, `.md`, `.txt`, and `.pdf`. [5]

MCP exposes the same Phase 1 direction through `load_documents(document_paths)` and a complete-pipeline wrapper, but those are file-path-oriented tools rather than upload-oriented API handlers. [6][7]

Answer: The likely backing path is `CompleteGraphRAGPipeline.process_document()` / `execute_complete_pipeline()` for `.pdf` and `.txt`, not the cross-modal orchestrator placeholder path.

### A3: Wiring safety

Directly wiring `/api/analyze` is not a trivial endpoint patch because the endpoint accepts uploaded bytes while `CompleteGraphRAGPipeline` and T01 take filesystem paths. A safe adapter would need to:

- persist the upload to a controlled temp path with cleanup on all success/failure paths;
- narrow or route supported formats, because the current T01 path handles `.pdf` and `.txt`, not every API-allowed extension;
- adapt complete-pipeline output to `AnalysisResponse` without inventing graph/table/vector results;
- define behavior when graph build/query dependencies are unavailable;
- add an integration-style test around a real tiny `.txt` or PDF fixture.

Answer: wiring is feasible but larger than the current API status-honesty slice. The next implementation should be a small `analyze_document_file_path(...)` adapter or a narrowed `/api/analyze` path for `.txt` first, with explicit acceptance criteria.

## Recommendation

Do not re-enable `/api/analyze` by calling the cross-modal orchestrator with placeholder graph data. The next safe implementation slice is:

1. Add a project-local adapter that takes a filesystem path plus task/format options and calls `CompleteGraphRAGPipeline.process_document(...)`.
2. Start with `.txt` only, because it avoids PDF fixture complexity and is supported by `T01PDFLoaderUnified`.
3. Return a deliberately narrow response shape with real extracted counts/stage outputs, or adjust `AnalysisResponse` to match the complete-pipeline result instead of forcing old cross-modal fields.
4. Add a current-runtime test that writes a temporary `.txt` file, runs the adapter, and proves no placeholder graph path is used.

Confidence: medium-high for path identification; medium for direct API wiring because graph build/query service readiness still needs runtime verification.

## Open Questions

- Does `CompleteGraphRAGPipeline` complete end-to-end in the isolated `.venv` without Neo4j or with the current default graph services?
- Should `/api/analyze` remain a high-level document-analysis endpoint, or should a new endpoint expose complete-pipeline execution with a response model that matches actual pipeline stages?
- Should `.docx`, `.doc`, and `.md` stay in the API validation list only after dedicated loaders are wired?

## Assumptions Register

| # | Assumption | Confidence | How to verify | Round | Status |
|---|---|---|---|---|---|
| 1 | `CompleteGraphRAGPipeline` is the intended real end-to-end document path. | Medium-high | Run a tiny `.txt` fixture through `process_document()` and inspect stage output. | 1 | Open |
| 2 | Starting with `.txt` is safer than PDF for the first API adapter test. | High | Compare fixture/setup complexity and T01 support. | 1 | Open |
| 3 | Current graph build/query dependencies may block full pipeline execution. | Medium | Run the tiny fixture in the isolated `.venv` and capture the first real failure. | 1 | Open |

## Files Consulted

- [1] `src/api/cross_modal_api.py`
- [2] `tests/current_runtime/test_cross_modal_api_contract.py`
- [3] `src/analytics/complete_pipeline.py`
- [4] `src/analytics/complete_pipeline.py`
- [5] `src/tools/phase1/t01_pdf_loader_unified.py`
- [6] `src/tools/phase1/phase1_mcp_tools.py`
- [7] `src/mcp_tools/pipeline_tools.py`
