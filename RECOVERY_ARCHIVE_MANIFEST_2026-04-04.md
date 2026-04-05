# Recovery Archive Manifest

This manifest records the local full-record preservation work done on April 4, 2026 for the KGAS / Digimons lineage at [phd_thesis_work](/home/brian/projects/phd_thesis_work).

## Intent

The tracked `kgas1` checkout restored the clean repository state, but it did not include the full local filesystem record that had accumulated around the project. To avoid losing anything that may have been intentionally or unintentionally removed from the tracked repo, a non-tracked archive layer now exists at `archive_full_record/`.

That layer is ignored by git on purpose. It is a preservation area, not part of the clean working tree.

## Preserved Material

The following sources are preserved under `archive_full_record/`:

| Archive Path | Preservation Mode | Original Source | Approx Size | Notes |
| --- | --- | --- | --- | --- |
| `archive_full_record/filesystem_snapshots/phd_thesis_work_filesystem_snapshot_20260404_192641` | moved snapshot | `/home/brian/projects/phd_thesis_work_archives_backup_20260404_192641` | 301M | The exact archive-only directory that existed before the clean repo restore |
| `archive_full_record/lineage_variants/Digimons_minimal` | symlink | `/home/brian/projects/archive/Digimons_minimal` | 49M | Local repo matching `kgas1` `origin/master` commit `2c59a1f` |
| `archive_full_record/lineage_variants/Digimons_clean_for_real` | symlink | `/home/brian/projects/archive/Digimons_clean_for_real` | 77M | Local KGAS snapshot with cleanup-focused branch history |
| `archive_full_record/lineage_variants/digimons_old` | symlink | `/home/brian/projects/archive/digimons_old` | 6.0G | Large pre-KGAS / earlier Digimons archive |
| `archive_full_record/lineage_variants/digimon_core_sparse` | symlink | `/home/brian/projects/archive/digimon_core_sparse` | 26M | Sparse / contract-oriented variant |
| `archive_full_record/lineage_variants/Digimons_docs` | symlink | `/home/brian/projects/archive/Digimons_docs` | 14M | Documentation extraction variant |
| `archive_full_record/lineage_variants/digimon_lineage_Digimons` | symlink | `/home/brian/projects/archive/digimon_lineage_Digimons` | 9.0G | Largest preserved lineage bundle |
| `archive_full_record/lineage_variants/digimon-v2` | symlink | `/home/brian/projects/archive/digimon-v2` | 1.3G | Later Digimon lineage repo |
| `archive_full_record/lineage_variants/digimon_autoloop` | symlink | `/home/brian/projects/archive/digimon_autoloop` | 94M | Autoloop-related lineage variant |

## Restored Into The Clean Repo

These files were restored into the tracked repo itself because they appeared to be specifically missed from the working path:

- `experimental/experimental_files/main.py`
- `experimental/experimental_files/minimal_mcp_server.py`
- `experimental/experimental_files/minimal_working_mcp.py`
- `experimental/experimental_files/proper_mcp_server.py`
- `experimental/experimental_files/simple_fastmcp_server.py`
- `experimental/experimental_files/simple_mcp_server.py`
- `experimental/experimental_files/working_mcp_server.py`

## Rationale

- Keep the clean repo usable.
- Preserve the full surviving local record under the expected project path.
- Avoid immediately reintroducing tens of gigabytes of archived lineage material into tracked git history.
- Make later curation possible without another recovery pass.

## Next Step

If a clean filtered version is needed, curate from `archive_full_record/` into tracked paths deliberately, one recovery slice at a time.
