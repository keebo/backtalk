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
arithmetic produced as next-token generation. This finds a plain binary
operation in the question with regex, computes it with Python's own
arithmetic (never eval() on arbitrary text — operands and the operator
are extracted first, THIS module does the actual math), and hands Rosa
the answer to state back.

Best-effort and narrow by design: one operation per question, digits
only (no spelled-out numbers), the common spoken phrasings. Anything
outside that is left for Rosa to handle — or decline — herself, same as
before this module existed.
"""
import re

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


def resolve(text: str) -> list[tuple[str, str]]:
    """[(matched phrase, formatted result), ...] for the first
    arithmetic expression found in text. At most one match — a spoken
    question asks one thing at a time, and matching more risks
    resolving fragments of the same expression twice."""
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
