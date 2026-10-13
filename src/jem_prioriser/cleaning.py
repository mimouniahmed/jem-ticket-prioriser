"""Shared cleaning rules: automated accounts to exclude and priority classes.

- Bots are identified by their e-mail domain; they produce 96% of P4 (wptsync) and 92% of P5
  (intermittent-bug-filer) bugs and are excluded.
- Without bots, P4 is almost unused (~900 bugs since 2020): it is merged with P5 into a
  "Low" class ("will not be fixed by Mozilla").
"""

from __future__ import annotations

# 13 accounts checked by hand, all automated (intermittent-bug-filer, wptsync, bugmon, update-bot,
# clouseau-bot...). @disabled.tld accounts are deactivated human users and are kept.
BOT_DOMAINS = ("@mozilla.bugs", "@mozilla.tld", "@bmo.tld", "@bots.tld")

# Bugzilla priority -> model class (ordered from most to least urgent)
LABEL_OF_PRIORITY = {"P1": "P1", "P2": "P2", "P3": "P3", "P4": "Low", "P5": "Low"}
CLASSES = ["P1", "P2", "P3", "Low"]


def is_bot(email: str) -> bool:
    return email.endswith(BOT_DOMAINS)


def human_sql(column: str = "creator") -> str:
    """SQL condition excluding bots, to be used in a WHERE clause."""
    return " AND ".join(f"{column} NOT LIKE '%{domain}'" for domain in BOT_DOMAINS)
