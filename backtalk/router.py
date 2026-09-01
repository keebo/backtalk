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
"""Local-vs-cloud routing for backtalk's dispatch loop.

The point isn't task "difficulty" — it's whether a question needs
anything Claude alone has: tools, vault access, or memory of this
conversation. Every turn that reaches Claude reloads the full system
prompt and tool set regardless of how simple the question is, so a
self-contained question (a quick fact, a bit of math, rephrasing a
sentence) costs real tokens for no reason.

Deliberately rule-based, not model-judged: letting the small local
model decide its own routing is an easy way for it to quietly misroute
something that actually needed Claude. A keyword miss is at least
visible in the log and easy for Kevin to tune by editing the list
below — a bad judgment call by a 7B model is neither.

Default leans LOCAL, not cloud: unmatched text routes local. This is
safe specifically because the voice/face indicator (see main.py's
speak_reply_local and signals.set_source) always tells Kevin which one
answered, and the force-cloud phrase below is the escape hatch for
"no, I want the real answer" without more than half a second of talking.
"""

# Any of these appearing routes to Claude. Substring match, lowercase,
# deliberately broad — biased toward over-routing to cloud rather than
# under-routing, since a mistaken cloud turn just costs some tokens
# while a mistaken local turn risks a wrong answer read as YOUR answer.
CLOUD_KEYWORDS = (
    # memory / conversation history
    "remember", "recall", "earlier", "yesterday", "last time",
    "we talked", "we discussed", "you said", "you told me",
    # tool / action verbs
    "check", "look up", "look into", "find ", "search",
    "email", "send ", "message", "text ",
    "schedule", "calendar", "meeting", "appointment",
    "commit", "push ", "git ", "code", "fix ", "bug", "file ", "edit",
    "run ", "execute", "restart", "open ", "close ",
    "remind me", "task", "todo", "to-do",
    # this system and Kevin's own tools/data
    "vault", "note", "daily note", "active priorit",
    "wrike", "hedy", "gmail", "calendar", "drive",
    "backtalk", "visualizer", "kokoro", "cipher", "cypher",
    "routine", "trigger", "p2p", "bravespan",
    # URLs and links: Rosa has no internet access at all, so a link is
    # ALWAYS a cloud question, unconditionally — confirmed missed live
    # 2026-08-31 when a pasted GitHub URL fell through (no other keyword
    # matched) and Rosa hallucinated a description of the repo from its
    # name alone instead of declining.
    "http", "www.",
)

# Spoken to force a question to Claude regardless of what the keyword
# scan would have decided — the explicit override Kevin asked for.
# Matched as a PREFIX after normalization, so it's stripped cleanly
# off the front of the utterance.
FORCE_CLOUD_PHRASES = (
    "ask cipher directly",
    "ask cypher directly",
    "ask the real cipher",
    "ask the real cypher",
)

# The reverse: opening an utterance with the local model's own name
# forces it there regardless of keywords — Kevin's explicit ask.
# Read from config, not hardcoded, so a future rename (local_llm.name)
# doesn't leave a stale trigger behind. If Whisper ever mishears this
# name the way it mishears "Cipher" (see backtalk.json's quit_phrases
# for that precedent), add the misheard spellings here too.
def _force_local_names() -> tuple[str, ...]:
    from backtalk.config import CFG
    name = CFG.get("local_llm", {}).get("name") or ""
    return (name.lower(),) if name else ()


def _norm(text: str) -> str:
    """Local copy of main.py's _norm_speech normalization (lowercase,
    non-letters to spaces, collapsed) — duplicated rather than imported
    to avoid a circular import between router.py and main.py."""
    out = []
    for ch in text.lower():
        out.append(ch if ch.isalpha() else " ")
    return " ".join("".join(out).split())


def strip_force_cloud(text: str) -> tuple[str, bool]:
    """Returns (text_to_use, forced). If a force-cloud phrase opens the
    utterance, it's stripped and forced=True; otherwise the text is
    returned unchanged with forced=False."""
    norm = _norm(text)
    for phrase in FORCE_CLOUD_PHRASES:
        if norm.startswith(phrase):
            # Strip the same number of words from the front of the
            # ORIGINAL text (not the normalized one) so punctuation in
            # the remainder survives.
            word_count = len(phrase.split())
            remainder = " ".join(text.split()[word_count:]).strip(" ,.-")
            return (remainder or text), True
    return text, False


def strip_force_local(text: str) -> tuple[str, bool]:
    """Returns (text_to_use, forced). If the utterance opens by naming
    the local model directly ("Rosa, ..."), it's stripped and
    forced=True; otherwise the text is returned unchanged with
    forced=False. Checked before FORCE_CLOUD_PHRASES/route() in
    main.py — addressing her by name is the most explicit signal
    there is, so it wins over everything else, including a pending
    awaiting_answer flag."""
    norm = _norm(text)
    for name in _force_local_names():
        if norm == name or norm.startswith(name + " "):
            remainder = " ".join(text.split()[1:]).strip(" ,.-")
            return (remainder or text), True
    return text, False


# Proofread/tone-edit mode — Kevin's ask 2026-08-31. Unconditional,
# like strip_force_local: this always means "run local_llm.edit() on
# whatever follows," regardless of keywords or a pending awaiting_answer
# state, since it's an explicit, deliberate request for a specific
# capability rather than an ordinary question that needs routing.
PROOFREAD_PHRASES = ("proofread this",)


def strip_proofread(text: str) -> tuple[str, bool]:
    """Returns (text_to_edit, triggered). Prefix-matched like the other
    force phrases — "proofread this" opens the utterance, everything
    after it (the instruction, if any, plus the text itself) is handed
    to local_llm.edit() untouched."""
    norm = _norm(text)
    for phrase in PROOFREAD_PHRASES:
        if norm.startswith(phrase):
            word_count = len(phrase.split())
            remainder = " ".join(text.split()[word_count:]).strip(" ,.-:")
            return (remainder or text), True
    return text, False


def route(text: str) -> str:
    """"local" or "cloud" — see module docstring for the reasoning."""
    low = text.lower()
    for kw in CLOUD_KEYWORDS:
        if kw in low:
            return "cloud"
    return "local"


# Phrases that ask Kevin for a decision without necessarily ending the
# sentence in "?" — confirmed missed live 2026-08-31: "Still waiting on
# your yes to actually build that one." is exactly this shape, and the
# bare-"?" check let Kevin's "Yes, build it" fall through to the local
# router's default instead of forcing back to whoever asked.
_AWAITING_CUES = (
    "waiting on your", "want me to", "should i", "need your",
    "let me know", "your call", "go ahead and confirm",
    # "I'm expecting you to hand me something next" phrasings — missed
    # live 2026-08-31: "Ready when you paste it" didn't match any cue
    # above, so the paste that followed fell through to the router
    # instead of forcing back to whoever asked for it.
    "when you paste", "go ahead and paste", "paste it", "share it",
    "send it over",
)


def is_awaiting_answer(sentence: str) -> bool:
    """True if this (the last spoken sentence of a reply) is the kind
    of thing that expects Kevin's next utterance to answer it — either
    a literal question, or one of the common non-"?" phrasings that
    still asks for a decision."""
    s = sentence.rstrip()
    if s.endswith("?"):
        return True
    low = s.lower()
    return any(cue in low for cue in _AWAITING_CUES)
