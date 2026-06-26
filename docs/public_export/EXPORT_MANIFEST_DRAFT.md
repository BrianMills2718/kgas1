# Export Manifest Draft

Date: 2026-06-26
Status: Draft only; no export has been built, approved, or published.

## Export Goal

Create a reviewable documentation-first derivative that preserves the intellectual and engineering record of the PhD/KGAS work while excluding raw preservation artifacts and local operational data.

## Proposed Included Paths

| Path | Rationale | Review Status |
| --- | --- | --- |
| `README.md` | Public overview of KGAS framing. | Needs final human read. |
| `RECOVERY_ARCHIVE_MANIFEST_2026-04-04.md` | Explains the preservation archive without embedding the archive. | Needs final human read. |
| `docs/` | Current docs, plans, architecture, public-export policy, and API references. | Needs final human read. |
| `investigations/` | Focused investigation and review artifacts. | Needs final human read. |
| `thesis_record_wiki/wiki/` | Karpathy-style synthesized wiki pages, concepts, sources, and log. | Needs final human read. |
| `thesis_record_wiki/raw/source_manifest.md` | Source/hash manifest for provenance. | Needs final human read. |

Approximate current sizes:

- `thesis_record_wiki/wiki/`: 1.6M, 197 files.
- `docs/`: 608K.
- `investigations/`: 60K.

## Excluded Paths By Default

| Path or Pattern | Reason |
| --- | --- |
| `.env` | Local credentials/configuration. |
| `.git/` | Repository internals and remote metadata. |
| `.venv/` | Local dependency environment. |
| `archive_full_record/` | Raw preservation archive; may contain logs, backups, credentials, datasets, and historical clutter. |
| `data/` | Local databases and runtime state, including `data/provenance.db`. |
| `logs/` | Full operational logs, including `logs/super_digimon.log`. |
| `exports/` | Generated output area; must not be recursively republished. |
| `*.db`, `*.sqlite`, `*.sqlite3` | Runtime databases. |
| `*.log` | Operational logs. |
| `*.bundle` | Git recovery bundles. |
| `*.tar`, `*.tar.gz`, `*.zip` | Archives may contain unreviewed nested files. |
| `*.pem`, `*.key`, `*.crt`, `*.p12` | Credential-bearing material. |

## Known Local Risk Items

The current repo root contains these risk items that must remain excluded from any public/export candidate:

- `.env`
- `data/provenance.db`
- `logs/super_digimon.log`
- `logs/super_digimon.rotating.log`
- ignored `archive_full_record/`

## Required Scan Commands

Run these against the export candidate directory, not the raw repo:

```bash
find "$EXPORT_DIR" -type f | sort > "$EXPORT_DIR.inventory.txt"
du -sh "$EXPORT_DIR" > "$EXPORT_DIR.size.txt"
rg -n --hidden --glob '!.git' --glob '!*.png' --glob '!*.jpg' --glob '!*.jpeg' --glob '!*.pdf' \
  '(api[_-]?key|secret|token|password|authorization|bearer|BEGIN (RSA|OPENSSH|PRIVATE) KEY|sk-[A-Za-z0-9_-]{20,})' \
  "$EXPORT_DIR" > "$EXPORT_DIR.secret-scan.txt"
```

The scan output must be reviewed before publication. Placeholder examples are allowed only when clearly non-live and documented.

## Publication Decision

Default destination: private GitHub repository.

Public release requires Brian's explicit approval after reviewing the final inventory and scan outputs.
