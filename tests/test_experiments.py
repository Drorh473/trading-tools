import json

from backtest import experiments


def _arm(name="baseline", n=100, gross=0.10, net=0.05, drop3=0.02, win=0.5):
    return {
        "name": name, "n": n, "gross_expectancy": gross, "net_expectancy": net,
        "drop_top3_expectancy": drop3, "win_rate": win,
    }


def test_record_experiment_writes_a_file_and_returns_its_path(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, "git_sha", lambda cwd=None: "sha123")

    path = experiments.record_experiment(
        "btc_trend_gate",
        hypothesis="a BTC 200MA regime gate improves net expectancy",
        universe="35 watchlist symbols",
        fit_period="year 1",
        confirm_period="year 2",
        arms=[_arm("baseline"), _arm("gated", net=0.08)],
        verdict="reverted",
        experiments_dir=str(tmp_path),
    )

    assert path.startswith(str(tmp_path))
    data = json.loads(open(path).read())
    assert data["name"] == "btc_trend_gate"
    assert data["verdict"] == "reverted"
    assert data["git_sha"] == "sha123"
    assert "written_at" in data


def test_record_experiment_requires_drop_top3_on_every_arm(tmp_path):
    """The registry actively encodes the reading discipline (see
    feedback-sweep-past-the-optimum) - an arm missing drop_top3_expectancy
    is refused rather than silently recorded as a result nobody can trust
    without re-deriving that number by hand."""
    bad_arm = {"name": "baseline", "n": 100, "gross_expectancy": 0.1, "net_expectancy": 0.05}

    try:
        experiments.record_experiment(
            "x", hypothesis="h", universe="u", fit_period="y1", confirm_period=None,
            arms=[bad_arm], verdict="null", experiments_dir=str(tmp_path),
        )
        assert False, "expected a ValueError"
    except ValueError as exc:
        assert "drop_top3_expectancy" in str(exc)


def test_two_runs_of_the_same_named_experiment_do_not_overwrite_each_other(tmp_path, monkeypatch):
    """The BTC trend gate's own postmortem: 'the original arms came from two
    different runs' - a registry that let a second run clobber the first
    would make that exact conflation the DEFAULT behaviour instead of the
    mistake it was."""
    monkeypatch.setattr(experiments, "git_sha", lambda cwd=None: "sha1")
    first = experiments.record_experiment(
        "x", hypothesis="h", universe="u", fit_period="y1", confirm_period=None,
        arms=[_arm()], verdict="null", experiments_dir=str(tmp_path),
    )
    monkeypatch.setattr(experiments, "git_sha", lambda cwd=None: "sha2")
    second = experiments.record_experiment(
        "x", hypothesis="h", universe="u", fit_period="y1", confirm_period=None,
        arms=[_arm()], verdict="null", experiments_dir=str(tmp_path),
    )

    assert first != second
    import os
    assert os.path.exists(first)
    assert os.path.exists(second)


def test_list_experiments_reads_every_recorded_run_sorted_by_time(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, "git_sha", lambda cwd=None: "sha")
    experiments.record_experiment(
        "first", hypothesis="h", universe="u", fit_period="y1", confirm_period=None,
        arms=[_arm()], verdict="null", experiments_dir=str(tmp_path),
    )
    experiments.record_experiment(
        "second", hypothesis="h", universe="u", fit_period="y1", confirm_period=None,
        arms=[_arm()], verdict="confirmed", experiments_dir=str(tmp_path),
    )

    runs = experiments.list_experiments(str(tmp_path))

    assert [r["name"] for r in runs] == ["first", "second"]


def test_list_experiments_on_an_empty_directory_returns_an_empty_list(tmp_path):
    assert experiments.list_experiments(str(tmp_path)) == []
