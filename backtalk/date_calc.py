# backtalk: talk to your Claude Code agent out loud.
# Copyright (C) 2026 Jared Rhodenizer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Deterministic date-relative phrase resolution, for Rosa specifically.

Rosa is a small local model answering with no tool-calling loop — asking
her to do date arithmetic in her head is exactly the kind of thing a
small LLM gets wrong under pressure (confirmed 2026-08-31: she answered
"what was the date on Friday" three days off). No model size guarantees
100% on arithmetic done as next-token generation, so this module does the
math with plain datetime instead: it finds date-relative phrases with
regex, resolves them deterministically, and hands Rosa the already-solved
fact to state back — no arithmetic left for her to get wrong.

Best-effort by design: anything not matched here is left for Rosa to
handle (or decline) herself, same as before this module existed.
"""
import re
from datetime import date, timedelta

_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday",
             "saturday", "sunday"]

_WEEKDAY_RE = re.compile(
    r'\b(next|last|this)?\s*(' + '|'.join(_WEEKDAYS) + r')\b', re.IGNORECASE)

_RELATIVE_DAY_RE = re.compile(r'\b(today|tomorrow|yesterday)\b', re.IGNORECASE)

_WEEK_RE = re.compile(r'\b(next|last)\s+week\b', re.IGNORECASE)

# Two alternatives sharing one regex: "in/next N days/weeks" (future) and
# "N days/weeks ago" (past) — kept as one pattern so resolve() only has
# one offset-shaped case to branch on.
_OFFSET_RE = re.compile(
    r'\b(in|next)\s+(\d+)\s+(day|days|week|weeks)\b'
    r'|\b(\d+)\s+(day|days|week|weeks)\s+ago\b', re.IGNORECASE)

# A bare weekday with no next/last/this qualifier is ambiguous ("what's
# the date Friday?") — lean on the sentence's own tense as the tiebreaker
# rather than always guessing future.
_PAST_CUE_RE = re.compile(r'\b(was|were|did|had)\b', re.IGNORECASE)


def _nearest_weekday(today: date, target_idx: int, direction: str) -> date:
    """direction: "next" (strictly future, not today), "last" (strictly
    past, not today), "this" (within the Mon-Sun week containing today,
    may be past or future), or "auto" (nearest future occurrence, not
    today — the fallback for an unqualified weekday with no past-tense
    cue in the sentence)."""
    if direction == "next":
        delta = (target_idx - today.weekday()) % 7
        return today + timedelta(days=delta or 7)
    if direction == "last":
        back = (today.weekday() - target_idx) % 7
        return today - timedelta(days=back or 7)
    if direction == "this":
        monday = today - timedelta(days=today.weekday())
        return monday + timedelta(days=target_idx)
    delta = (target_idx - today.weekday()) % 7          # auto
    return today + timedelta(days=delta or 7)


def resolve(text: str, today: date) -> list[tuple[str, date]]:
    """[(matched phrase, resolved date), ...] for every date-relative
    phrase found in text, relative to today."""
    found = []

    for m in _RELATIVE_DAY_RE.finditer(text):
        word = m.group(1).lower()
        d = {"today": today, "tomorrow": today + timedelta(days=1),
             "yesterday": today - timedelta(days=1)}[word]
        found.append((m.group(0), d))

    for m in _WEEK_RE.finditer(text):
        d = today + timedelta(days=7 if m.group(1).lower() == "next" else -7)
        found.append((m.group(0), d))

    for m in _OFFSET_RE.finditer(text):
        if m.group(2):      # "in/next N days/weeks" — future
            n, unit = int(m.group(2)), m.group(3).lower()
            d = today + timedelta(days=n * (7 if unit.startswith("week") else 1))
        else:                # "N days/weeks ago" — past
            n, unit = int(m.group(4)), m.group(5).lower()
            d = today - timedelta(days=n * (7 if unit.startswith("week") else 1))
        found.append((m.group(0), d))

    for m in _WEEKDAY_RE.finditer(text):
        qualifier = (m.group(1) or "").lower()
        target_idx = _WEEKDAYS.index(m.group(2).lower())
        direction = qualifier if qualifier in ("next", "last", "this") else (
            "last" if _PAST_CUE_RE.search(text) else "auto")
        found.append((m.group(0), _nearest_weekday(today, target_idx, direction)))

    return found


def annotate(text: str, today: date) -> str:
    """A short factual note for Rosa's prompt so a date-relative question
    is answered by stating a precomputed fact, not by computing one.
    Empty string if nothing in the text matched."""
    matches = resolve(text, today)
    if not matches:
        return ""
    facts = "; ".join(f'"{phrase}" is {d.strftime("%A, %B %-d, %Y")}'
                       for phrase, d in matches)
    return ("Resolved dates for this question — state these directly, "
            f"do not recompute them: {facts}.")
