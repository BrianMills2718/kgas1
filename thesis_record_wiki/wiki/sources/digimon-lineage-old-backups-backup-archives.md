---
type: Source
title: Digimon Lineage Old Backups Backup Archives
description: Three tar.gz backup archives and backup_history metadata from old_backups_2025_08, with duplicate-ID and unencrypted .env caveats.
tags: [source, old-backups, backups, archives, security]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/old_backups_2025_08/backups/
confidence: high
---

# Summary

`archive/old_backups_2025_08/backups/` contains three `tar.gz` backups plus `backup_history.json`, 208 KB total, aggregate SHA-256 `9c20731526529cbc3ab877c6157186d6e3ff42e890c2c78ff95dae2743628548`.

The history records all four backup events as successful, compressed, and unencrypted. The tarballs were listed during ingest but not extracted or modified. [1]

# Backup History

| Backup ID | Type | Timestamp | Size | Status | Encryption |
| --- | --- | --- | ---: | --- | --- |
| `backup_20250717_234023` | full | 2025-07-17T23:40:23 | 187,258 | success | false |
| `backup_20250717_234023` | incremental | 2025-07-17T23:40:23 | 187,344 | success | false |
| `backup_20250718_013211` | full | 2025-07-18T01:32:11 | 1,091 | success | false |
| `backup_20250718_021743` | full | 2025-07-18T02:17:43 | 11,504 | success | false |

# Contents Observed

Tarball listings show:

- `backup_20250717_234023.tar.gz`: logs, Neo4j directory, config files including `.env`, and results files.
- `backup_20250718_013211.tar.gz`: logs and Neo4j directory.
- `backup_20250718_021743.tar.gz`: logs, Neo4j directory, config files, contract schema, environment compose files, examples, and monitoring config.

# Caveats

The history contains a duplicate `backup_id` and file path for the 2025-07-17 full and incremental records, with different checksums. The actual tarball hash on disk is `5510bb8aa0639f37ebcc232ac743e17eb3685a0257ec53b8ae4164c050eb673f`, matching the incremental record rather than the earlier full-record checksum. [1]

At least one tarball contains `config/.env`, and the history says encryption is false. Treat these backup archives as potentially containing secrets. Do not publish or extract them into a public workspace without a secret-scrubbing plan.

# Interpretation

This slice is backup-system evidence: backup metadata existed, backups were generated quickly, and archive contents covered logs/config/results. It is not a guarantee of complete recoverability. The duplicate-ID/path issue and unencrypted `.env` content are both preservation risks to keep visible.

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/old_backups_2025_08/backups/backup_history.json`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/old_backups_2025_08/backups/`
