# Public Export Candidate Review

Date: 2026-06-26
Candidate: `exports/public_candidate_20260626_094828/`
Status: Local review complete; not approved for publication.

## Candidate Contents

The temporary candidate was built from the documentation-first include list:

- `README.md`
- `RECOVERY_ARCHIVE_MANIFEST_2026-04-04.md`
- `docs/`
- `investigations/`
- `thesis_record_wiki/wiki/`
- `thesis_record_wiki/raw/source_manifest.md`

The candidate is local and ignored by git under `exports/`. It was not published or copied to a remote.

## Size And Inventory

- Candidate size: 2.4M.
- File count: 251 files.
- Secret-pattern scan hits: 74 lines.
- Forbidden-file check found no `.env`, database, log, git bundle, compressed archive, private-key, or certificate files in the candidate.

Generated local review files:

- `exports/public_candidate_20260626_094828.inventory.txt`
- `exports/public_candidate_20260626_094828.size.txt`
- `exports/public_candidate_20260626_094828.secret-scan.txt`

These generated files are local ignored artifacts and are not committed.

## Scan Finding Summary

The 74 secret-pattern hits are concentrated in documentation and wiki pages that discuss credentials, demo passwords, tokens, API-key placeholders, and public-export policy. The scan did not find a raw `.env`, database, log file, bundle, compressed archive, private key, or certificate in the candidate.

Review-needed categories:

| Category | Examples | Disposition |
| --- | --- | --- |
| Demo Neo4j passwords | `password`, `newpassword`, `testpassword` in setup/how-to docs and wiki summaries. | Review before publication; likely acceptable only if clearly labeled demo values. |
| API key placeholders | `your_openai_key`, `YOUR_SERP_KEY`, `GOOGLE_API_KEY=your-api-key-here`. | Review before publication; likely acceptable as placeholders but should be normalized if public. |
| Security-policy text | public-export docs, plan docs, and wiki pages mentioning secrets/tokens. | Expected false positives; keep because they explain the boundary. |
| Historical risk summaries | wiki/source-manifest entries saying raw archives contain `.env`, logs, or possible credentials without reproducing values. | Expected and useful; keep unless Brian wants a less security-detailed public narrative. |
| Credential-presence claims | historical validation pages mention credential variables were present, without values. | Review before publication; not a raw secret but may reveal operational context. |

## Current Publication Decision

Do not publish this candidate yet.

The candidate is much safer than the raw repository/archive because it excludes the obvious high-risk file classes, but it still needs Brian's human review for narrative/privacy posture. The key open question is whether public readers should see the detailed security-risk summaries about old raw archives, or whether those should be moved to a private/internal appendix.

## Safe Completion Result

This satisfies the safe organization export gate:

- a docs-only candidate can be built without touching the raw archive;
- the candidate excludes the known high-risk raw paths and file types;
- scan results are recorded and bounded;
- publication remains an explicit approval gate.

## Remaining Approval Gates

- Brian approval required before publishing any export candidate.
- Brian approval required before making a private or public GitHub repository for the export.
- Brian approval required before including any raw archive-derived file beyond synthesized wiki/source-manifest text.
- Brian approval required before redacting or rewriting any candidate text for public narrative choices.
