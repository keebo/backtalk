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
"""Deterministic arithmetic resolution, for Rosa specifically — the
general-math sibling of date_calc.py.

Same reasoning as that module: no LLM, of any size, guarantees 100% on
arithmetic produced as next-token generation. This finds either a plain
binary operation OR a statistic over a whole list of numbers in the
question with regex, computes it with Python's own arithmetic (never
eval() on arbitrary text — operands are extracted first, THIS module
does the actual math), and hands Rosa the answer to state back.

Extended 2026-09-08 after a real, confirmed miss: asked to sum an
18-number list read aloud, Rosa (reasoning through the addition herself,
since the original single-binary-op version below only ever caught one
operation like "5 plus 3") got a different, wrong total. List-statistics
(sum, average/mean, median, standard deviation, variance, minimum,
maximum, count) are resolved separately and checked FIRST, so a long
list is never misread as just its first two numbers by the older
single-op patterns below.

Best-effort and narrow by design: digits only (no spelled-out numbers),
the common spoken phrasings. Anything outside that is left for Rosa to
handle — or decline — herself, same as before this module existed.
"""
import re
import statistics

_NUM = r'-?\d+(?:\.\d+)?'


def _num(s: str) -> float:
    return float(s)


# Order matters: percent-of is checked first since it shares "of" with
# nothing else here, then the four basic operators. Each entry is
# (regex, operand order in the match, function).
_PERCENT_OF_RE = re.compile(
    rf'({_NUM})\s*(?:%|percent)\s*of\s*({_NUM})', re.IGNORECASE)

_SUBTRACTED_FROM_RE = re.compile(
    rf'({_NUM})\s*subtracted from\s*({_NUM})', re.IGNORECASE)

_OPS = [
    (re.compile(rf'({_NUM})\s*(?:\+|plus|added to)\s*({_NUM})', re.IGNORECASE),
     lambda a, b: a + b, "+"),
    (re.compile(rf'({_NUM})\s*(?:-|minus)\s*({_NUM})', re.IGNORECASE),
     lambda a, b: a - b, "-"),
    (re.compile(rf'({_NUM})\s*(?:\*|x|times|multiplied by)\s*({_NUM})', re.IGNORECASE),
     lambda a, b: a * b, "*"),
    (re.compile(rf'({_NUM})\s*(?:/|divided by)\s*({_NUM})', re.IGNORECASE),
     lambda a, b: a / b if b else None, "/"),
]


def _fmt(n: float) -> str:
    if float(n).is_integer():
        return str(int(n))
    return f"{n:.4f}".rstrip("0").rstrip(".")


# (trigger regex, human label, function taking the number list). Order
# doesn't matter here -- unlike the single-op list below, these are
# independent keywords that can't overlap in the same question the way
# "+" could shadow "-" -- so every one that matches gets computed, not
# just the first (Kevin's ask: "sum and average" in one question should
# answer both, not pick one).
_STAT_TRIGGERS = [
    (re.compile(r'\bsum\b|\btotal\b', re.IGNORECASE), "sum", sum),
    (re.compile(r'\baverage\b|\bmean\b', re.IGNORECASE), "average",
     statistics.mean),
    (re.compile(r'\bmedian\b', re.IGNORECASE), "median", statistics.median),
    (re.compile(r'\bstandard deviation\b|\bstd\.?\s*dev\b', re.IGNORECASE),
     "standard deviation", statistics.stdev),
    (re.compile(r'\bvariance\b', re.IGNORECASE), "variance",
     statistics.variance),
    (re.compile(r'\b(?:minimum|smallest|lowest)\b', re.IGNORECASE),
     "minimum", min),
    (re.compile(r'\b(?:maximum|largest|highest)\b', re.IGNORECASE),
     "maximum", max),
    (re.compile(r'\bhow many numbers\b|\bcount\b', re.IGNORECASE),
     "count", len),
]

# Below this, the numbers found are already fully covered by the
# single-binary-op path further down ("5 plus 3") -- no reason for list
# mode to compete with that for a plain two-number question.
_LIST_MIN_NUMBERS = 2


def _resolve_list_stats(text: str) -> list[tuple[str, str]]:
    """Statistics over a whole LIST of numbers pulled out of the
    question -- sum, average, median, standard deviation, variance,
    minimum, maximum, count. Checked first in resolve(), so a long
    list read aloud is never misread as just its first two numbers by
    the older single-op patterns below (that miss is exactly what
    prompted building this)."""
    numbers = [float(n) for n in re.findall(_NUM, text)]
    if len(numbers) < _LIST_MIN_NUMBERS:
        return []
    results = []
    for pattern, label, fn in _STAT_TRIGGERS:
        if not pattern.search(text):
            continue
        if fn in (statistics.stdev, statistics.variance) and len(numbers) < 2:
            continue  # needs at least 2 data points, not just "a list"
        try:
            value = fn(numbers)
        except statistics.StatisticsError:
            continue
        phrase = f"the {label} of those {len(numbers)} numbers"
        results.append((phrase, _fmt(value)))
    return results


def resolve(text: str) -> list[tuple[str, str]]:
    """[(matched phrase, formatted result), ...] for either a list
    statistic (see _resolve_list_stats — can return more than one, e.g.
    "sum and average" in the same question) or, failing that, the
    first single binary arithmetic expression found in text. At most
    one match in the single-op case — a spoken question asks one thing
    at a time there, and matching more risks resolving fragments of
    the same expression twice."""
    list_stats = _resolve_list_stats(text)
    if list_stats:
        return list_stats

    m = _PERCENT_OF_RE.search(text)
    if m:
        pct, base = _num(m.group(1)), _num(m.group(2))
        return [(m.group(0), _fmt(pct / 100 * base))]

    m = _SUBTRACTED_FROM_RE.search(text)
    if m:
        a, b = _num(m.group(1)), _num(m.group(2))
        return [(m.group(0), _fmt(b - a))]

    for pattern, fn, _sym in _OPS:
        m = pattern.search(text)
        if m:
            a, b = _num(m.group(1)), _num(m.group(2))
            result = fn(a, b)
            if result is None:      # division by zero
                continue
            return [(m.group(0), _fmt(result))]

    return []


def annotate(text: str) -> str:
    """A short factual note for Rosa's prompt so an arithmetic question
    is answered by stating a precomputed result, not by computing one.
    Empty string if nothing in the text matched."""
    matches = resolve(text)
    if not matches:
        return ""
    facts = "; ".join(f'"{phrase}" is {result}' for phrase, result in matches)
    return ("Resolved calculations for this question — state these "
            f"directly, do not recompute them: {facts}.")
