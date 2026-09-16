"""Tests for build/build_github_snapshot.py. No live GitHub API calls."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from build import build_github_snapshot as snap  # noqa: E402


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, text="", headers=None):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text
        self.headers = headers or {}

    def json(self):
        return self._json_data


def make_repo(name, **overrides):
    repo = {
        "name": name,
        "owner": {"login": "leonarduk"},
        "description": f"{name} description",
        "html_url": f"https://github.com/leonarduk/{name}",
        "topics": ["agents"],
        "stargazers_count": 3,
        "pushed_at": "2026-08-20T10:00:00Z",
        "fork": False,
        "archived": False,
        "private": False,
    }
    repo.update(overrides)
    return repo


class TestFetchRepos:
    def test_filters_forks_archived_and_private(self, monkeypatch):
        repos = [
            make_repo("keep-me"),
            make_repo("a-fork", fork=True),
            make_repo("an-archive", archived=True),
            make_repo("a-private-repo", private=True),
        ]

        def fake_get(url, headers, params, timeout):
            if params["page"] == 1:
                return FakeResponse(200, json_data=repos)
            return FakeResponse(200, json_data=[])

        monkeypatch.setattr(snap.requests, "get", fake_get)

        result = snap.fetch_repos("leonarduk")

        assert [r["name"] for r in result] == ["keep-me"]

    def test_paginates_until_short_page(self, monkeypatch):
        page1 = [make_repo(f"repo-{i}") for i in range(100)]
        page2 = [make_repo("repo-100")]
        calls = []

        def fake_get(url, headers, params, timeout):
            calls.append(params["page"])
            if params["page"] == 1:
                return FakeResponse(200, json_data=page1)
            if params["page"] == 2:
                return FakeResponse(200, json_data=page2)
            return FakeResponse(200, json_data=[])

        monkeypatch.setattr(snap.requests, "get", fake_get)

        result = snap.fetch_repos("leonarduk")

        assert len(result) == 101
        assert calls == [1, 2]

    def test_rate_limit_raises(self, monkeypatch):
        def fake_get(url, headers, params, timeout):
            return FakeResponse(
                403, json_data={}, headers={"X-RateLimit-Remaining": "0"}
            )

        monkeypatch.setattr(snap.requests, "get", fake_get)

        with pytest.raises(snap.GitHubRateLimitError):
            snap.fetch_repos("leonarduk")

    def test_other_api_error_raises(self, monkeypatch):
        def fake_get(url, headers, params, timeout):
            return FakeResponse(500, json_data={})

        monkeypatch.setattr(snap.requests, "get", fake_get)

        with pytest.raises(snap.GitHubAPIError):
            snap.fetch_repos("leonarduk")


class TestFetchLanguages:
    def test_orders_by_bytes_descending(self, monkeypatch):
        def fake_get(url, headers, params, timeout):
            return FakeResponse(
                200, json_data={"Python": 100, "Shell": 500, "Dockerfile": 100}
            )

        monkeypatch.setattr(snap.requests, "get", fake_get)

        result = snap.fetch_languages("leonarduk", "some-repo")

        assert result == ["Shell", "Dockerfile", "Python"]

    def test_missing_languages_returns_empty_list(self, monkeypatch):
        def fake_get(url, headers, params, timeout):
            return FakeResponse(404, json_data={})

        monkeypatch.setattr(snap.requests, "get", fake_get)

        assert snap.fetch_languages("leonarduk", "some-repo") == []


class TestFetchReadme:
    def test_returns_raw_text(self, monkeypatch):
        def fake_get(url, headers, params, timeout):
            return FakeResponse(200, text="# Title\n\nBody text.")

        monkeypatch.setattr(snap.requests, "get", fake_get)

        assert snap.fetch_readme("leonarduk", "some-repo") == "# Title\n\nBody text."

    def test_missing_readme_returns_empty_string(self, monkeypatch):
        def fake_get(url, headers, params, timeout):
            return FakeResponse(404, text="")

        monkeypatch.setattr(snap.requests, "get", fake_get)

        assert snap.fetch_readme("leonarduk", "some-repo") == ""


class TestStripMarkdown:
    def test_strips_badges_links_headings_emphasis(self):
        text = (
            "![build](https://img.shields.io/badge.svg)\n"
            "# Title\n\n"
            "**Bold** and `code` and [a link](https://example.com).\n"
            "<div>raw html</div>\n"
        )
        result = snap.strip_markdown(text)
        assert "img.shields.io" not in result
        assert "#" not in result
        assert "**" not in result
        assert "<div>" not in result
        assert "a link" in result

    def test_truncates_to_limit(self):
        long_text = "word " * 1000
        excerpt = snap.readme_excerpt(long_text, limit=50)
        assert len(excerpt) == 50

    @pytest.mark.parametrize(
        "identifier",
        [
            "GITHUB_TOKEN",
            "multi_agent_coder",
            "other_var_names",
        ],
    )
    def test_does_not_mangle_snake_case_identifiers(self, identifier):
        result = snap.strip_markdown(f"Set the {identifier} environment variable.")
        assert identifier in result


class TestParseProjectsMd:
    def test_parses_sections_into_notes(self, tmp_path):
        path = tmp_path / "projects.md"
        path.write_text(
            "## issue-worm\n\nA multi-agent coder.\nSecond line.\n\n## allotmint\n\nA portfolio app.\n",
            encoding="utf-8",
        )

        notes = snap.parse_projects_md(path)

        assert notes["issue-worm"] == "A multi-agent coder.\nSecond line."
        assert notes["allotmint"] == "A portfolio app."

    def test_missing_file_returns_empty_dict(self, tmp_path):
        assert snap.parse_projects_md(tmp_path / "does-not-exist.md") == {}


class TestPrivacyGuardrail:
    """The avatar's knowledge base must only ever contain public information
    (docs/design.md §3.2, §6) — a private repo reaching github.json would be
    a real disclosure, not a cosmetic bug. This exercises the full
    build_snapshot() -> fetch_repos() path against the raw HTTP layer,
    rather than mocking fetch_repos itself, so a future regression in the
    private-repo filter would actually be caught here."""

    def test_private_repo_never_reaches_the_final_snapshot(self, monkeypatch, tmp_path):
        repos = [
            make_repo("public-repo"),
            make_repo("private-repo", private=True),
        ]

        def fake_get(url, headers, params=None, timeout=None):
            if url.endswith("/repos") and params["page"] == 1:
                return FakeResponse(200, json_data=repos)
            if url.endswith("/repos"):
                return FakeResponse(200, json_data=[])
            if url.endswith("/languages"):
                return FakeResponse(200, json_data={})
            if url.endswith("/readme"):
                return FakeResponse(200, text="")
            raise AssertionError(f"unexpected URL in test: {url}")

        monkeypatch.setattr(snap.requests, "get", fake_get)

        records = snap.build_snapshot(
            "leonarduk", projects_md_path=tmp_path / "projects.md"
        )

        names = [r["name"] for r in records]
        assert names == ["public-repo"]
        assert "private-repo" not in names


class TestBuildSnapshot:
    def _patch_all(self, monkeypatch, repos, languages=None, readme=""):
        monkeypatch.setattr(snap, "fetch_repos", lambda user, token=None: repos)
        monkeypatch.setattr(
            snap, "fetch_languages", lambda owner, name, token=None: languages or []
        )
        monkeypatch.setattr(
            snap, "fetch_readme", lambda owner, name, token=None: readme
        )

    def test_sorted_by_name(self, monkeypatch, tmp_path):
        repos = [make_repo("zeta"), make_repo("alpha")]
        self._patch_all(monkeypatch, repos)
        empty_projects_md = tmp_path / "projects.md"

        records = snap.build_snapshot("leonarduk", projects_md_path=empty_projects_md)

        assert [r["name"] for r in records] == ["alpha", "zeta"]

    def test_written_json_has_sorted_keys(self, monkeypatch, tmp_path):
        repos = [make_repo("alpha")]
        self._patch_all(monkeypatch, repos)
        records = snap.build_snapshot(
            "leonarduk", projects_md_path=tmp_path / "projects.md"
        )
        out_path = tmp_path / "github.json"

        snap.write_snapshot(records, out_path)

        raw_object_keys = list(
            json.loads(out_path.read_text(encoding="utf-8"))[0].keys()
        )
        assert raw_object_keys == sorted(raw_object_keys)

    def test_curated_note_merges_onto_matching_repo(self, monkeypatch, tmp_path):
        repos = [make_repo("issue-worm"), make_repo("other-repo")]
        self._patch_all(monkeypatch, repos)
        projects_md = tmp_path / "projects.md"
        projects_md.write_text(
            "## issue-worm\n\nA multi-agent coder.\n", encoding="utf-8"
        )

        records = snap.build_snapshot("leonarduk", projects_md_path=projects_md)

        by_name = {r["name"]: r for r in records}
        assert by_name["issue-worm"]["curated_note"] == "A multi-agent coder."
        assert by_name["other-repo"]["curated_note"] is None

    def test_missing_readme_and_no_topics_does_not_crash(self, monkeypatch, tmp_path):
        repos = [make_repo("bare-repo", topics=[], description=None)]
        self._patch_all(monkeypatch, repos, readme="")

        records = snap.build_snapshot(
            "leonarduk", projects_md_path=tmp_path / "projects.md"
        )

        assert records[0]["topics"] == []
        assert records[0]["description"] == ""
        assert records[0]["readme_excerpt"] == ""

    def test_pushed_at_truncated_to_date(self, monkeypatch, tmp_path):
        repos = [make_repo("some-repo", pushed_at="2026-08-20T10:30:00Z")]
        self._patch_all(monkeypatch, repos)

        records = snap.build_snapshot(
            "leonarduk", projects_md_path=tmp_path / "projects.md"
        )

        assert records[0]["pushed_at"] == "2026-08-20"

    def test_deterministic_output(self, monkeypatch, tmp_path):
        """build_snapshot() must produce byte-identical output on repeated
        runs with the same input. This exercises the full build_snapshot()
        path against the raw HTTP layer (like TestPrivacyGuardrail) so that
        a regression in sorting or dict ordering would actually be caught.

        The mocked dataset deliberately uses repo names whose alphabetical
        order differs from insertion order, and languages whose byte counts
        tie, to expose any non-deterministic ordering."""
        repos = [
            make_repo("zeta-repo"),
            make_repo("alpha-repo"),
            make_repo("mid-repo"),
            make_repo("beta-repo"),
        ]
        languages = {"Python": 100, "Shell": 500, "Dockerfile": 100}

        def fake_get(url, headers, params=None, timeout=None):
            if url.endswith("/repos") and params["page"] == 1:
                return FakeResponse(200, json_data=repos)
            if url.endswith("/repos"):
                return FakeResponse(200, json_data=[])
            if url.endswith("/languages"):
                return FakeResponse(200, json_data=languages)
            if url.endswith("/readme"):
                return FakeResponse(200, text="# Title\n\nBody text.")
            raise AssertionError(f"unexpected URL in test: {url}")

        monkeypatch.setattr(snap.requests, "get", fake_get)

        projects_md = tmp_path / "projects.md"
        projects_md.write_text(
            "## alpha-repo\n\nAlpha note.\n\n## zeta-repo\n\nZeta note.\n",
            encoding="utf-8",
        )

        first = snap.build_snapshot("leonarduk", projects_md_path=projects_md)
        second = snap.build_snapshot("leonarduk", projects_md_path=projects_md)

        assert first == second

        first_path = tmp_path / "first.json"
        second_path = tmp_path / "second.json"
        snap.write_snapshot(first, first_path)
        snap.write_snapshot(second, second_path)

        assert first_path.read_bytes() == second_path.read_bytes()


class TestWriteSnapshot:
    def test_deterministic_across_two_runs(self, tmp_path):
        records = [{"name": "b"}, {"name": "a"}]
        out_path = tmp_path / "github.json"

        snap.write_snapshot(records, out_path)
        first = out_path.read_text(encoding="utf-8")
        snap.write_snapshot(records, out_path)
        second = out_path.read_text(encoding="utf-8")

        assert first == second

    def test_never_writes_a_token(self, tmp_path):
        records = [{"name": "a", "curated_note": None}]
        out_path = tmp_path / "github.json"

        snap.write_snapshot(records, out_path)

        content = out_path.read_text(encoding="utf-8")
        assert "ghp_" not in content
        parsed = json.loads(content)
        assert parsed == records


class TestCli:
    def test_rate_limit_error_exits_nonzero_without_writing(
        self, monkeypatch, tmp_path
    ):
        def raise_rate_limit(user, token=None):
            raise snap.GitHubRateLimitError("boom")

        monkeypatch.setattr(snap, "build_snapshot", raise_rate_limit)
        out_path = tmp_path / "github.json"

        exit_code = snap.main(["--user", "leonarduk", "--out", str(out_path)])

        assert exit_code == 1
        assert not out_path.exists()

    def test_happy_path_writes_file(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            snap, "build_snapshot", lambda user, token=None: [{"name": "a"}]
        )
        out_path = tmp_path / "github.json"

        exit_code = snap.main(["--user", "leonarduk", "--out", str(out_path)])

        assert exit_code == 0
        assert out_path.exists()
