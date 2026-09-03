"""Renders the monthly report as Telegram-friendly markdown.

The shape is deliberate: three verdict lines at the top, everything below is
evidence for them. A month where all three read clean should be readable in
five seconds and closed. A report that has to be studied to find out whether
anything is wrong is a report that eventually stops being opened - the same
failure that let the weekly report go missing for two weeks.

Numbers that could not be produced say so. There is no path here that prints
a zero standing in for "not available": a $0.00 funding line and a funding
line that never ran look identical to a reader, and only one of them is true.
"""

from monthly_review.analyze import Autonomy, MonthlyReport, Reconciliation
from monthly_review.cadence import Cadence
from monthly_review.noise import Finding


def render(report: MonthlyReport) -> str:
    lines = [f"# Monthly Review — {report.label}", ""]
    lines += _verdict(report)
    lines.append("")
    lines += _balance(report.reconciliation)
    lines.append("")
    lines += _failures(report)
    lines.append("")
    lines += _tuning(report)
    lines.append("")
    lines += _autonomy(report.autonomy)
    return "\n".join(lines)


def _verdict(report: MonthlyReport) -> list[str]:
    """The whole report in three lines. Read these; open the rest only if one
    of them is not clean."""
    rec = report.reconciliation
    failures, fired = report.failures, report.fired

    if rec.actual is None:
        balance = "Balance:  not available — see below"
    else:
        pct = (rec.actual / rec.equity_start * 100) if rec.equity_start else 0.0
        balance = (
            f"Balance:  ${rec.equity_start:.2f} → ${rec.equity_end:.2f} "
            f"({pct:+.1f}%, {rec.actual:+.2f})"
        )

    fee_flag = 1 if (report.fee_model and report.fee_model.fires) else 0
    return [
        "```",
        balance,
        f"Failures: {len(failures)} needing attention",
        f"Tuning:   {len(fired) + fee_flag} flag(s) outside the noise band",
        "```",
    ]


def _balance(rec: Reconciliation) -> list[str]:
    lines = ["## Balance", ""]
    if rec.equity_start is None:
        lines += [
            f"**No opening balance: {rec.equity_start_note or 'unknown reason'}.**",
            "",
            "Bitget has no endpoint for past equity — it has to be recorded as",
            "it happens, at the month boundary, or the change being shown covers",
            "a different window than the fees and funding beside it. This run has",
            "written a fresh snapshot, so next month's report will have one.",
            "",
        ]
    lines += [
        f"- Realized P&L (closed trades): {rec.realized_pnl:+.2f}",
        f"- Trading fees: {_money(rec.fees)}",
        f"- Funding: {_money(rec.funding)} (positive = paid out)",
    ]
    if rec.residual is None:
        lines.append("- Reconciliation: not available without both an equity snapshot and a client")
        return lines

    lines += [
        f"- Explained by the bot: {rec.explained:+.2f}",
        f"- Actual equity change: {rec.actual:+.2f}",
        f"- **Residual: {rec.residual:+.2f}**",
    ]
    if abs(rec.residual) < 0.01:
        lines.append("")
        lines.append("Everything that moved the balance is accounted for.")
    elif rec.residual_is_explainable:
        lines.append("")
        lines.append(
            f"{rec.open_at_end} position(s) were open at the snapshot, so unrealized P&L "
            "sits in that residual legitimately."
        )
    else:
        lines.append("")
        lines.append(
            "**Books were flat and the residual is not zero.** Something moved the "
            "balance that the bot did not model. This is a defect, not a statistic."
        )
    return lines


def _failures(report: MonthlyReport) -> list[str]:
    lines = ["## Failures", ""]
    failures = report.failures
    if not failures:
        lines.append(f"All {len(report.cadence)} live instances signalled within their cadence.")
        return lines

    for c in failures:
        lines.append(f"- {_cadence_line(c)}")
    lines.append("")
    lines.append(
        "Thresholds are each instance's own (notifier.main.signal_silence_days), "
        "so this never disagrees with the scanner about what 'too quiet' means — "
        "only the window does. The scanner sees a snapshot; this sees the month."
    )
    return lines


def _cadence_line(c: Cadence) -> str:
    if c.silent_all_month:
        return f"**{c.tag}** — SILENT ALL MONTH. Zero signals in {c.longest_silence_days:.0f} days."
    return (
        f"**{c.tag}** — {c.breaches} silence(s) past its {c.threshold_days:.0f}-day threshold; "
        f"longest {c.longest_silence_days:.1f} days ({c.signals} signals)"
    )


def _tuning(report: MonthlyReport) -> list[str]:
    lines = ["## Tuning", ""]
    fired = report.fired
    fee = report.fee_model

    if fee and fee.fires:
        lines += [
            f"- **Fee model is off by {(fee.ratio - 1) * 100:+.0f}%** — modelled "
            f"${fee.modelled:.2f}, actually paid ${fee.actual:.2f}. Every strategy's "
            "sizing is built on the modelled number.",
        ]
    for finding in fired:
        lines.append(f"- **{finding.metric}**: {finding.live:+.2f} vs {finding.baseline:+.2f} "
                     f"over n={finding.n} (band ±{finding.detectable:.2f})")

    if not fired and not (fee and fee.fires):
        lines.append("Nothing outside its noise band.")

    quiet = [f for f in report.slippage.values() if not f.fires]
    if quiet:
        lines += ["", "Measured, not flagged — with the smallest difference each could have seen:", ""]
        for finding in sorted(quiet, key=lambda f: f.metric):
            lines.append(f"- {finding.metric}: {_quiet_line(finding)}")
    return lines


def _quiet_line(f: Finding) -> str:
    if f.detectable == float("inf"):
        return f"{f.live:+.2f} over n={f.n} — {f.note}"
    return f"{f.live:+.2f} (n={f.n}); nothing bigger than ±{f.detectable:.2f} would have shown"


def _autonomy(a: Autonomy) -> list[str]:
    lines = ["## Autonomy", ""]
    if not a.total_signals:
        lines.append("No signals dispatched this month.")
        return lines

    lines += [
        f"- Signals: {a.total_signals} "
        f"({a.approved} approved, {a.rejected} rejected, {a.never_acted_on} never acted on)",
        f"- Needed a human decision: {a.intervention_rate:.0%}",
        f"- Trades whose stop or target you changed after the proposal: {a.changed_from_plan}",
        "",
        "The goal is for the first two numbers to fall. Approving counts as an "
        "intervention: a bot that runs alone does not wait to be told.",
    ]
    return lines


def _money(value: float | None) -> str:
    return "not available" if value is None else f"{value:+.2f}"
