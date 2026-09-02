from backtest import experiments
from tools.experiments import format_table


def _arm(name="baseline", net=0.05):
    return {"name": name, "n": 100, "gross_expectancy": 0.1, "net_expectancy": net, "drop_top3_expectancy": 0.02}


def test_format_table_shows_name_verdict_and_each_arms_net_expectancy(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, "git_sha", lambda cwd=None: "sha")
    experiments.record_experiment(
        "btc_trend_gate", hypothesis="h", universe="u", fit_period="y1", confirm_period="y2",
        arms=[_arm("baseline", 0.05), _arm("gated", 0.08)], verdict="reverted",
        experiments_dir=str(tmp_path),
    )

    text = format_table(experiments.list_experiments(str(tmp_path)))

    assert "btc_trend_gate" in text
    assert "reverted" in text
    assert "baseline=+0.050R" in text
    assert "gated=+0.080R" in text


def test_format_table_on_no_runs_says_so_plainly(tmp_path):
    assert "no experiments" in format_table(experiments.list_experiments(str(tmp_path)))
