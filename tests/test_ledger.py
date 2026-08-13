"""The ledger exists because three features never worked and nothing noticed.

The partial take-profit failed every time for five months; the weekly report
crashed before sending every Sunday for two weeks; the untracked-position
dedupe forgot on every restart. Each was found by Dror noticing, and each was
invisible because a capability that has never once worked looks exactly like a
capability with nothing to do.

So the case these tests care most about is not "is it late" - it is "has it
ever happened at all", which is the case that hid for five months.
"""

from datetime import datetime, timedelta, timezone

from core import ledger


def _db(tmp_path):
    return str(tmp_path / "trades.db")


def test_a_capability_that_has_never_worked_is_reported_once_it_has_had_the_chance(tmp_path):
    """No record is not an absence of information - it IS the information.

    Treating it as "nothing to report" is precisely how the take-profit stayed
    broken from 2026-08-03 to 2026-08-13 across five symbols.
    """
    db = _db(tmp_path)
    now = datetime(2026, 8, 13, tzinfo=timezone.utc)
    ledger.began_watching(db, now=now - timedelta(days=30))

    statuses = ledger.survey(db, {ledger.TAKE_PROFIT_PLACED: 7.0}, now=now)

    assert len(statuses) == 1
    assert statuses[0].never is True
    assert statuses[0].alarming is True
    assert "has NEVER worked" in ledger.format_survey(statuses)


def test_a_freshly_deployed_ledger_does_not_declare_everything_broken(tmp_path):
    """The first report after any deploy would otherwise carry a NEVER line for
    every watched capability, none of which has had a chance to work yet. An
    alert that noisy trains you to ignore it, which is the same failure as the
    five-minute Telegram expiry - and it would have arrived attached to the one
    report that is actually read."""
    db = _db(tmp_path)
    now = datetime(2026, 8, 13, tzinfo=timezone.utc)
    ledger.began_watching(db, now=now - timedelta(days=2))

    statuses = ledger.survey(db, {
        ledger.TAKE_PROFIT_PLACED: 14.0,
        ledger.ENTRY_ORDER_PLACED: 7.0,
    }, now=now)

    assert all(s.never for s in statuses), "genuinely no record yet"
    assert not any(s.alarming for s in statuses), "but not yet news"
    assert ledger.format_survey(statuses) == "All watched capabilities have worked recently."


def test_watching_begins_on_first_use_and_does_not_move(tmp_path):
    """If the start moved every time it was read, the grace period would never
    expire and a never-working capability would stay silent forever."""
    db = _db(tmp_path)
    first = ledger.began_watching(db, now=datetime(2026, 8, 1, tzinfo=timezone.utc))
    again = ledger.began_watching(db, now=datetime(2026, 9, 1, tzinfo=timezone.utc))

    assert first == again


def test_a_capability_is_watched_before_it_has_ever_succeeded(tmp_path):
    """The names are declared up front rather than created on first success.

    A ledger that only learns a capability exists when it works cannot ever
    report that it has not worked, which would reproduce the original bug in
    the tool built to catch it.
    """
    statuses = ledger.survey(_db(tmp_path), {ledger.TAKE_PROFIT_PLACED: 7.0})

    assert statuses[0].capability == ledger.TAKE_PROFIT_PLACED
    assert statuses[0].never is True
    assert not ledger.ever(_db(tmp_path), ledger.TAKE_PROFIT_PLACED)


def test_recording_a_success_makes_it_visible(tmp_path):
    db = _db(tmp_path)
    ledger.record(db, ledger.TAKE_PROFIT_PLACED)

    assert ledger.ever(db, ledger.TAKE_PROFIT_PLACED)
    assert ledger.survey(db, {ledger.TAKE_PROFIT_PLACED: 7.0})[0].alarming is False


def test_a_capability_that_worked_but_not_lately_is_overdue(tmp_path):
    db = _db(tmp_path)
    now = datetime(2026, 8, 13, tzinfo=timezone.utc)
    ledger.record(db, ledger.WEEKLY_REPORT, now=now - timedelta(days=9))

    status = ledger.survey(db, {ledger.WEEKLY_REPORT: 8.0}, now=now)[0]

    assert status.never is False
    assert status.overdue is True
    assert status.age == 9.0
    assert "9.0 days ago" in ledger.format_survey([status])


def test_working_within_its_window_is_not_reported(tmp_path):
    db = _db(tmp_path)
    now = datetime(2026, 8, 13, tzinfo=timezone.utc)
    ledger.record(db, ledger.WEEKLY_REPORT, now=now - timedelta(days=6))

    statuses = ledger.survey(db, {ledger.WEEKLY_REPORT: 8.0}, now=now)

    assert statuses[0].alarming is False
    assert ledger.format_survey(statuses) == "All watched capabilities have worked recently."


def test_never_worked_sorts_above_merely_late(tmp_path):
    """Five months of a silently broken take-profit is a different and worse
    thing than a report that is a day late, and the report should say so
    first."""
    db = _db(tmp_path)
    now = datetime(2026, 8, 13, tzinfo=timezone.utc)
    ledger.record(db, ledger.WEEKLY_REPORT, now=now - timedelta(days=30))

    ledger.began_watching(db, now=now - timedelta(days=60))

    statuses = ledger.survey(db, {
        ledger.WEEKLY_REPORT: 8.0,
        ledger.TAKE_PROFIT_PLACED: 7.0,
    }, now=now)

    assert statuses[0].capability == ledger.TAKE_PROFIT_PLACED
    assert statuses[0].never is True


def test_a_strategy_tag_with_slashes_and_spaces_survives_as_a_filename(tmp_path):
    """"Strategy 3 1D/5m" is a real tag and a real path separator. Recording it
    must not create a directory called "Strategy 3 1D"."""
    db = _db(tmp_path)
    capability = ledger.signal_seen("Strategy 3 1D/5m")

    ledger.record(db, capability)

    assert ledger.ever(db, capability)
    written = [p for p in (tmp_path / "ledger").iterdir() if p.name != ".watch_started"]
    assert len(written) == 1
    assert written[0].is_file()


def test_two_capabilities_do_not_share_a_file(tmp_path):
    """The scanner and the weekly cron are different processes. A shared
    document would be a read-modify-write race that could lose exactly the
    update this module exists to keep."""
    db = _db(tmp_path)
    now = datetime(2026, 8, 13, tzinfo=timezone.utc)

    ledger.record(db, ledger.signal_seen("Strategy 1 1H"), now=now)
    ledger.record(db, ledger.signal_seen("Strategy 1 4H"), now=now - timedelta(days=3))

    assert ledger.age_days(db, ledger.signal_seen("Strategy 1 1H"), now=now) == 0.0
    assert ledger.age_days(db, ledger.signal_seen("Strategy 1 4H"), now=now) == 3.0


def test_an_unreadable_record_reads_as_never_rather_than_crashing(tmp_path):
    """A corrupt file must degrade to "no evidence it worked", which is the
    safe direction: it over-reports rather than going quiet."""
    db = _db(tmp_path)
    ledger.record(db, ledger.TAKE_PROFIT_PLACED)
    path = next(p for p in (tmp_path / "ledger").iterdir() if p.name != ".watch_started")
    path.write_text("not a timestamp", encoding="utf-8")

    assert ledger.last_success(db, ledger.TAKE_PROFIT_PLACED) is None
    assert ledger.survey(db, {ledger.TAKE_PROFIT_PLACED: 7.0})[0].never is True
