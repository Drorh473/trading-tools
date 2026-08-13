"""The portfolio backtest, and the only instrument this project has for
answering "does the whole thing make money".

It lived in a session scratchpad under AppData/Local/Temp until 2026-08-13,
where Windows could delete it at any time and had already cost one rebuild
from scratch. Committing it is not tidiness: every strategy decision made from
here on cites a number this package produced, and a number nobody can
reproduce is not evidence.

  python -m backtest.run [n_symbols] [hours] [workers]

engine     - the account: fees, the $5 per-leg floor, sizing, fills, exits
portfolio  - signal generation (parallel) and portfolio replay (sequential)
run        - the CLI, and the arms that get compared
"""
