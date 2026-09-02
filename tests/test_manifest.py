import json
import subprocess

from backtest import manifest


def test_write_manifest_records_a_git_sha_and_timestamp(tmp_path, monkeypatch):
    monkeypatch.setattr(manifest, "git_sha", lambda cwd=None: "deadbeef" * 5)
    out = str(tmp_path / "signals.pkl")

    manifest.write_manifest(out)

    data = json.loads((tmp_path / "signals.pkl.manifest.json").read_text())
    assert data["git_sha"] == "deadbeef" * 5
    assert "written_at" in data


def test_write_manifest_includes_caller_supplied_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(manifest, "git_sha", lambda cwd=None: "sha")
    out = str(tmp_path / "signals.pkl")

    manifest.write_manifest(out, instance_count=3, universe=["BTCUSDT", "ETHUSDT"])

    data = json.loads((tmp_path / "signals.pkl.manifest.json").read_text())
    assert data["instance_count"] == 3
    assert data["universe"] == ["BTCUSDT", "ETHUSDT"]


def test_read_manifest_returns_what_was_written(tmp_path, monkeypatch):
    monkeypatch.setattr(manifest, "git_sha", lambda cwd=None: "sha")
    out = str(tmp_path / "signals.pkl")
    manifest.write_manifest(out, instance_count=3)

    data = manifest.read_manifest(out)

    assert data["instance_count"] == 3


def test_read_manifest_returns_none_when_none_exists(tmp_path):
    assert manifest.read_manifest(str(tmp_path / "nonexistent.pkl")) is None


def test_git_sha_returns_none_outside_a_git_repo(tmp_path):
    assert manifest.git_sha(cwd=str(tmp_path)) is None


def test_git_sha_returns_a_real_commit_hash_inside_this_repo():
    sha = manifest.git_sha()

    assert sha is not None
    assert len(sha) == 40
    assert all(c in "0123456789abcdef" for c in sha)
    # And it's genuinely the same answer `git` itself gives, not a guess.
    expected = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert sha == expected
