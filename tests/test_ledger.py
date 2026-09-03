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


def test_silence_thresholds_are_keyed_on_the_instances_own_base_timeframe():
    """One blanket 21 days was wrong in both directions at once.

    Measured over 549 logged signals: Strategy 1 1H has never been quiet more
    than 1.97 days, so three weeks of silence before anyone asked would have
    hidden a dead instance for a fortnight. Strategy 1 1D genuinely idles 8.15
    days and needs the room. The base timeframe is the LAST one named in the
    tag - "Strategy 3 1D/1H" reads trend off the daily and acts on the hour -
    which is the same convention SWING_TAGS uses.
    """
    from notifier.main import signal_silence_days

    assert signal_silence_days("Strategy 1 1H") == 4.0
    assert signal_silence_days("Strategy 1 4H") == 7.0
    assert signal_silence_days("Strategy 1 1D") == 14.0
    # Trend timeframe first, base last: these act on 1H and 5m, not on 1D.
    assert signal_silence_days("Strategy 3 1D/1H") == 4.0
    assert signal_silence_days("Strategy 3 1D/5m") == 2.0


def test_strategy_2_1_keeps_its_two_days_whatever_its_timeframe_says():
    """It prompts many times a day, so one quiet day is already wrong - and
    that holds for its 1H instance even though 1H otherwise means four days."""
    from notifier.main import V21_TAGS, signal_silence_days

    assert V21_TAGS, "the override is meaningless if the set is empty"
    for tag in V21_TAGS:
        assert signal_silence_days(tag) == 2.0


def test_an_unrecognised_tag_falls_back_rather_than_crashing():
    from notifier.main import signal_silence_days

    assert signal_silence_days("something hand typed") == 14.0


def test_the_trailing_stop_records_against_a_real_db_path(tmp_path):
    """try_record's first argument is the db path, and the trailing call was
    passing the capability into it - a TypeError, which try_record does not
    catch (it only guards OSError). The stop would have moved on the exchange
    and then the Telegram message would never have been sent.

    It never fired in three days of logs, so nothing was lost; the bug was
    simply armed and waiting for the first runner to qualify.
    """
    import inspect

    from notifier import trailing_stops

    src = inspect.getsource(trailing_stops.TrailingStopManager.poll)
    assert "ledger.try_record(ledger.TRAILING_STOP_MOVED)" not in src
    assert "ledger.try_record(self.storage.db_path, ledger.TRAILING_STOP_MOVED)" in src


def test_the_weekly_review_tells_the_ledger_it_ran():
    """The heartbeat file and the ledger are two different watchers, and only
    the heartbeat was written - so the ledger held WEEKLY_REPORT as "never
    worked" while the report ran every Sunday, and would have announced its own
    death inside itself on the eighth day."""
    import inspect

    from weekly_review import main as wr

    src = inspect.getsource(wr.main)
    assert "ledger.try_record(settings.trades_db_path, ledger.WEEKLY_REPORT)" in src
