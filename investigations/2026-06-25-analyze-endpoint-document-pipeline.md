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
| A4 | Why did the Neo4j-backed `.txt` probe create entities but no relationships/edges? | A2 | Answered |

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

### Runtime Probe: tiny `.txt` fixture

Command shape:

```text
Create a temporary `.txt` file containing:
Alice works for Acme Corporation. Bob founded Beta Labs in Seattle.

Instantiate CompleteGraphRAGPipeline() and call process_document(path).
```

Initial observed result:

```text
Neo4j connection failed: Unsupported authentication token, missing key `credentials`
Cannot create IdentityService without Neo4j connection
STATUS error
EXC_TYPE RuntimeError
EXC Neo4j connection required for IdentityService
```

Interpretation: the first real blocker is service initialization, before T01 document loading runs. `CompleteGraphRAGPipeline.__init__` constructs `T01PDFLoaderUnified`, which immediately reads `service_manager.identity_service`; `ServiceManager.identity_service` requires a live Neo4j-backed `IdentityService` and raises when the driver is unavailable. [3][5][8]

After configuring the existing local Neo4j Docker container through `NEO4J_PASSWORD`, the probe exposed and repaired several runtime contract drifts:

- `DistributedTransactionManager` had current methods (`begin_transaction`, `commit_all`, `rollback_all`) while analytics callers used legacy names (`begin_distributed_transaction`, `commit_distributed_transaction`, `rollback_distributed_transaction`, `record_operation`). Compatibility methods now support logical operation recording and commit/rollback for these callers. [10]
- `CompleteGraphRAGPipeline._execute_document_loading(...)` expected top-level `text_content`, while current T01 returns nested `data["document"]["text"]`. The adapter now reads the current nested shape. [3][5]
- `_execute_text_chunking(...)` sent old `source_ref`/`text_content` keys, while current T15 requires `document_ref`/`text`/`document_confidence`. The adapter now sends the current shape and reads `total_chunks`. [3]
- `_execute_entity_extraction(...)` read `mentions`, while current T23A returns `entities`. The adapter now accepts both. [3]
- Graph building expected `text` mention fields, while T23A emits `surface_form`; complete-pipeline now normalizes extracted entities before relationship extraction and graph building. [3]
- `GraphBuilder` treated zero relationships as fatal by invoking T34 with an empty list; it now treats node-only graph construction as successful and skips T34 when there are no relationships. [9]

The first Neo4j-backed `.txt` probe result:

```text
STATUS success
pipeline_result_keys ['document_loading', 'entity_extraction', 'graph_building', 'pagerank_calculation', 'query_execution', 'relationship_extraction', 'text_chunking']
pipeline_stats {'documents_processed': 1, 'chunks_created': 1, 'entities_extracted': 4, 'relationships_extracted': 0, 'graph_nodes_created': 4, 'graph_edges_created': 0, 'queries_answered': 3, 'pipeline_stages_completed': 7, 'real_operations_confirmed': True}
proof {'all_steps_executed': True, 'real_operations_confirmed': True, 'neo4j_integration_verified': False, 'end_to_end_success': False}
```

Interpretation: the real `.txt` pipeline now executes all stages and creates Neo4j nodes. The fixture does not produce relationships, so the pipeline correctly remains a partial semantic success: real operations are confirmed, but `end_to_end_success` stays false because edge/relationship validation is not satisfied.

### A4: relationship/edge proof repair

Direct T23A + T27 probing with the same text showed that current T27 can extract relationships when it receives entities grouped with the right chunk provenance:

```text
Alice works for Acme Corporation. Bob founded Beta Labs in Seattle.
T27 status success count 5
Alice WORKS_FOR Acme Corporation
Bob CREATED Seattle
```

The complete-pipeline miss was an adapter bug. `T23ASpacyNERUnified` emits current entity records with `chunk_ref`, while `CompleteGraphRAGPipeline._execute_relationship_extraction(...)` grouped mentions only by `source_ref`. That caused each real chunk to appear to have fewer than two entities, so T27 was skipped. The repair groups by `source_ref` or `chunk_ref`, preserving historical source-ref callers while accepting the current T23A shape. [3][5]

The same slice repaired a proof-wiring issue: `GraphBuilder` called `Neo4jDockerManager.execute_read_query(...)`, but the manager exposed only `execute_query(...)` and async query variants. Adding the async read-query compatibility alias lets graph validation and statistics use the existing secure query executor. `CompleteGraphRAGPipeline._validate_complete_pipeline(...)` also now returns top-level `neo4j_verified` for the proof payload and treats end-to-end semantic success as requiring at least one relationship and one edge. [3][9][11]

Current Neo4j-backed `.txt` probe result after the repair:

```text
STATUS success
pipeline_stats {'documents_processed': 1, 'chunks_created': 1, 'entities_extracted': 4, 'relationships_extracted': 5, 'graph_nodes_created': 4, 'graph_edges_created': 5, 'queries_answered': 3, 'pipeline_stages_completed': 7, 'real_operations_confirmed': True}
proof {'all_steps_executed': True, 'real_operations_confirmed': True, 'neo4j_integration_verified': True, 'end_to_end_success': True}
relationship_types {'WORKS_FOR': 1, 'CREATED': 1, 'RELATED_TO': 3}
```

Focused tests now cover the current T23A `chunk_ref` grouping, the Neo4j manager compatibility alias, and the live Neo4j smoke requires at least one relationship, one edge, `neo4j_integration_verified=True`, and `end_to_end_success=True`. [12][13][14]

### API wiring slice: `.txt` only

The first `/api/analyze` wiring slice is now deliberately narrow. The endpoint still validates target format, optimization level, and validation level, then dispatches only `.txt` uploads through `CompleteGraphRAGPipeline.process_document(...)`. The upload is written to a controlled temporary `.txt` path, passed to the file-path-oriented pipeline, and removed in a `finally` block. [1][3]

The response maps the complete-pipeline output into the existing `AnalysisResponse` shape without inventing cross-modal orchestration fields:

- `workflow_id`: complete-pipeline transaction ID
- `selected_mode`: `complete_graphrag_pipeline`
- `results`: real `pipeline_stats`, `pipeline_results`, and `proof_of_completion`
- `validation`: complete-pipeline validation payload
- `source_traceability`: original filename, `.txt`, and document ref from T01 loading

Previously accepted but unproven extensions (`.pdf`, `.docx`, `.doc`, `.md`) now return 501 from `/api/analyze` with an explicit "only for .txt" message. Unsupported extensions such as `.exe` still return 400. Focused API tests prove temp-file cleanup, pipeline dispatch, response serialization, and non-text 501 behavior. A skip-safe live API test also calls `analyze_document(...)` with a `.txt` upload when `NEO4J_PASSWORD` is configured and asserts relationships, graph edges, Neo4j proof, and end-to-end success. [1][12]

The live API test exposed a graph-validation concern: `_analyze_graph_connectivity(...)` used a `CALL { ... }` variable-length traversal that the current input validator blocked as a dangerous pattern. Allowing that trusted internal query made the live pipeline hang on the accumulated local graph, so the repair classified full connected-component analysis as not computed during request-time validation. Graph validation now uses a bounded node/relationship summary query, returns `connectivity_check="not_computed_unbounded_traversal_skipped"`, and leaves entity/edge verification as the Neo4j proof basis. [9][13]

### Query result quality repair

The complete pipeline originally used generic canned queries:

```text
What entities are mentioned in the document?
What relationships exist between entities?
Who works for which organizations?
```

T49's query entity extractor only searches around concrete entities it can extract from the query and look up in Neo4j. Empirical checks showed those generic questions yielded no query entities and therefore zero paths, even though the graph contained Alice, Acme Corporation, Bob, and Seattle. Entity-name queries such as `Alice` and `Bob` returned related-entity paths. [3]

The repair now derives default smoke queries from extracted entity names when the caller does not supply `test_queries`, and `queries_answered` counts only successful query executions with non-empty result sets. A live `.txt` probe now reports:

```text
queries_answered: 2
query Alice count 10
query Acme Corporation count 0
query Bob count 10
```

This does not make T49 a general natural-language QA system, but it stops treating empty generic query executions as answered questions and proves at least some graph query results are returned from the built graph. [3][12][14]

### T49 query entity extraction repair

The follow-up T49 audit found a regex-level extraction bug. The first capitalized-phrase pattern allowed lowercase spaces, so questions such as `Who is connected to Alice?` produced the candidate `Who is connected to` and missed `Alice`. Entity indicator patterns also ran with `re.IGNORECASE`, which allowed `How is Alice connected to Acme Corporation?` to produce the noisy candidate `How is Alice connected to Acme`. [15]

The repair limits the primary pattern to capitalized tokens and removes case-insensitive matching from the indicator patterns. Pattern-only tests now assert:

```text
Who is connected to Alice? -> Alice
How is Alice connected to Acme Corporation? -> Alice, Acme Corporation
```

A live T49/GraphQueryEngine probe with Neo4j configured now returns five results for `Who is connected to Alice?`, with the top related-entity result connecting Alice to Seattle. [15]

## Recommendation

Do not broaden `/api/analyze` by calling the cross-modal orchestrator with placeholder graph data. The current safe slice is `.txt` only and backed by `CompleteGraphRAGPipeline.process_document(...)`. The next safe implementation slice is either a live API-level `.txt` test with Neo4j credentials available or a similarly narrow `.pdf` acceptance fixture after PDF behavior is proven through T01 and the complete pipeline.

Confidence: high for path identification; high that the `.txt` complete-pipeline path executes real stages, extracts relationships, creates Neo4j edges, reports end-to-end proof when Neo4j is configured, and is now reachable through `/api/analyze` for `.txt` uploads.

## Open Questions

- Should `GraphQueryEngine` query-stat helpers get the same read-query compatibility audit, or is the current successful-but-zero-path query result sufficient for the first `/api/analyze` slice?
- Should complete-pipeline query smoke tests prefer relationship endpoints such as source/target pairs rather than single-entity related-neighbor queries?
- Should graph query ranking deduplicate repeated entities from accumulated smoke-test runs so result quality is not dominated by local duplicate nodes?
- Should there be an explicit non-Neo4j test service manager for document-loader and text-only adapter tests?
- Should `/api/analyze` keep using the generic `AnalysisResponse` wrapper, or should a new endpoint expose complete-pipeline execution with a response model that exactly matches actual pipeline stages?
- Should `.docx`, `.doc`, and `.md` stay in the API validation list only after dedicated loaders are wired?

## Assumptions Register

| # | Assumption | Confidence | How to verify | Round | Status |
|---|---|---|---|---|---|
| 1 | `CompleteGraphRAGPipeline` is the intended real end-to-end document path. | High | Run a tiny `.txt` fixture through `process_document()` and inspect stage output. | 1 | Verified for real-stage execution, relationship extraction, edge creation, and proof flags with Neo4j configured. |
| 2 | Starting with `.txt` is safer than PDF for the first API adapter test. | High | Compare fixture/setup complexity and T01 support. | 1 | Verified by passing `.txt` runtime smoke test. |
| 3 | Current graph build/query dependencies may block full pipeline execution. | High | Run the tiny fixture in the isolated `.venv` and capture the first real failure. | 1 | Resolved for entity and edge graph execution; query execution succeeds but returns zero discovered paths for the canned queries. |
| 4 | Zero relationships were caused by fixture weakness rather than adapter grouping. | Medium | Compare direct T23A+T27 probe with complete-pipeline grouping. | 2 | Wrong; direct T27 extracted relationships, while complete-pipeline grouped current T23A `chunk_ref` entities under `unknown`. |

## Files Consulted

- [1] `src/api/cross_modal_api.py`
- [2] `tests/current_runtime/test_cross_modal_api_contract.py`
- [3] `src/analytics/complete_pipeline.py`
- [4] `src/analytics/complete_pipeline.py`
- [5] `src/tools/phase1/t01_pdf_loader_unified.py`
- [6] `src/tools/phase1/phase1_mcp_tools.py`
- [7] `src/mcp_tools/pipeline_tools.py`
- [8] `src/core/service_manager.py`
- [9] `src/analytics/graph_builder.py`
- [10] `src/core/distributed_transaction_manager.py`
- [11] `src/core/neo4j_manager.py`
- [12] `tests/current_runtime/test_analysis_agent_t27_contract.py`
- [13] `tests/current_runtime/test_neo4j_manager_compat.py`
- [14] `tests/current_runtime/test_complete_pipeline_neo4j_runtime.py`
- [15] `src/tools/phase1/multihop_query/query_entity_extractor.py`
