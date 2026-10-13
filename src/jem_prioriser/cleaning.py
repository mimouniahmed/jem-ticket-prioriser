"""Règles de nettoyage partagées : comptes automatiques à exclure et classes de priorité.

- les bots se reconnaissent à leur domaine e-mail ; ils produisent 96 % des P4 (wptsync)
  et 92 % des P5 (intermittent-bug-filer) et sont exclus ;
- sans les bots, P4 est quasi inutilisée (~900 bugs depuis 2020) : elle est fusionnée avec
  P5 en une classe « Basse » (« ne sera pas corrigé par Mozilla »).
"""

from __future__ import annotations

# 13 comptes vérifiés à la main, tous automatiques (intermittent-bug-filer, wptsync, bugmon,
# update-bot, clouseau-bot...). Les comptes @disabled.tld sont des humains désactivés : on les garde.
BOT_DOMAINS = ("@mozilla.bugs", "@mozilla.tld", "@bmo.tld", "@bots.tld")

# Priorité Bugzilla -> classe du modèle (ordonnées de la plus à la moins urgente)
LABEL_OF_PRIORITY = {"P1": "P1", "P2": "P2", "P3": "P3", "P4": "Basse", "P5": "Basse"}
CLASSES = ["P1", "P2", "P3", "Basse"]


def is_bot(email: str) -> bool:
    return email.endswith(BOT_DOMAINS)


def human_sql(column: str = "creator") -> str:
    """Condition SQL qui exclut les bots, à insérer dans un WHERE."""
    return " AND ".join(f"{column} NOT LIKE '%{domain}'" for domain in BOT_DOMAINS)
