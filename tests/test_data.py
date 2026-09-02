from backtest import manifest
from tools.data import format_table, list_datasets


def test_list_datasets_finds_every_pkl_recursively(tmp_path):
    (tmp_path / "a.pkl").write_bytes(b"x")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.pkl").write_bytes(b"yy")

    infos = list_datasets(str(tmp_path))

    assert {i.path for i in infos} == {"a.pkl", "sub/b.pkl"}


def test_list_datasets_reports_the_real_file_size(tmp_path):
    (tmp_path / "a.pkl").write_bytes(b"x" * 12345)

    infos = list_datasets(str(tmp_path))

    assert infos[0].size_bytes == 12345


def test_list_datasets_attaches_a_manifest_when_one_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(manifest, "git_sha", lambda cwd=None: "abc123")
    pkl = tmp_path / "signals.pkl"
    pkl.write_bytes(b"data")
    manifest.write_manifest(str(pkl), instance_count=3)

    infos = list_datasets(str(tmp_path))

    assert infos[0].manifest is not None
    assert infos[0].manifest["instance_count"] == 3
    assert infos[0].manifest["git_sha"] == "abc123"


def test_list_datasets_reports_none_when_no_manifest_exists(tmp_path):
    (tmp_path / "orphan.pkl").write_bytes(b"data")

    infos = list_datasets(str(tmp_path))

    assert infos[0].manifest is None


def test_format_table_shows_size_and_git_sha(tmp_path, monkeypatch):
    monkeypatch.setattr(manifest, "git_sha", lambda cwd=None: "abc123def456")
    pkl = tmp_path / "signals.pkl"
    pkl.write_bytes(b"x" * 2_000_000)
    manifest.write_manifest(str(pkl))

    text = format_table(list_datasets(str(tmp_path)))

    assert "signals.pkl" in text
    assert "1.9" in text  # 2,000,000 bytes / 1024^2 = 1.9 MiB, not 2.0
    assert "abc123def4" in text  # a truncated sha is fine; the full one is not required


def test_format_table_flags_a_missing_manifest_plainly(tmp_path):
    (tmp_path / "orphan.pkl").write_bytes(b"x")

    text = format_table(list_datasets(str(tmp_path)))

    assert "no manifest" in text


def test_format_table_reports_an_empty_directory_without_raising(tmp_path):
    assert "no .pkl" in format_table(list_datasets(str(tmp_path)))
