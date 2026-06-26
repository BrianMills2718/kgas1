"""Regression coverage for repository-local markdown link checks."""

from pathlib import Path

from scripts.check_markdown_links import check_markdown_links


def test_absolute_wiki_links_resolve_to_thesis_record_wiki(tmp_path: Path) -> None:
    """The wiki uses /wiki/... links even though files live under thesis_record_wiki/."""
    repo_root = tmp_path
    wiki_page = repo_root / "thesis_record_wiki" / "wiki" / "concepts" / "example.md"
    wiki_page.parent.mkdir(parents=True)
    wiki_page.write_text("# Example\n", encoding="utf-8")

    source = repo_root / "thesis_record_wiki" / "wiki" / "index.md"
    source.write_text("[Example](/wiki/concepts/example.md)\n", encoding="utf-8")

    violations = check_markdown_links(["thesis_record_wiki/wiki"], repo_root)

    assert violations == []


def test_absolute_progress_link_resolves_to_wiki_progress(tmp_path: Path) -> None:
    """The wiki overview links to /PROGRESS.md as a wiki-root document."""
    repo_root = tmp_path
    progress = repo_root / "thesis_record_wiki" / "PROGRESS.md"
    progress.parent.mkdir(parents=True)
    progress.write_text("# Progress\n", encoding="utf-8")

    source = repo_root / "thesis_record_wiki" / "wiki" / "overview.md"
    source.parent.mkdir(parents=True)
    source.write_text("[Progress](/PROGRESS.md)\n", encoding="utf-8")

    violations = check_markdown_links(["thesis_record_wiki/wiki"], repo_root)

    assert violations == []


def test_historical_marker_allows_missing_local_targets(tmp_path: Path) -> None:
    """Historical docs can preserve stale links when explicitly marked."""
    source = tmp_path / "docs" / "historical.md"
    source.parent.mkdir(parents=True)
    source.write_text(
        "<!-- link-check: allow-missing-historical-targets -->\n"
        "[Missing historical target](old/removed.md)\n",
        encoding="utf-8",
    )

    violations = check_markdown_links(["docs"], tmp_path)

    assert violations == []
