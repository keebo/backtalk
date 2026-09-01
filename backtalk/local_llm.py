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
"""A local, on-device LLM for self-contained questions that don't need
Claude's tools, vault access, or conversation memory — MLX, tuned for
Apple Silicon's unified memory, so it shares the GPU cleanly with
Kokoro instead of fighting it for a discrete card that doesn't exist
on a Mac.

Opt-in: nothing in this module runs unless CFG["local_llm"]["enabled"]
is true. See router.py for the decision of WHEN to use this; this
module only knows how, not when.
"""
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from backtalk import date_calc, math_calc
from backtalk.config import CFG
from backtalk.vlog import log

_model = None
_tokenizer = None
_lock = threading.Lock()

# Kevin's own timezone, not whatever the machine happens to be set to —
# same zone daily_token_report.py already anchors on for the same reason.
_TZ = ZoneInfo("America/New_York")


def _today():
    """Computed fresh on every call, never cached — a stale date baked
    into a long-lived process would be worse than no date at all."""
    return datetime.now(_TZ).date()


def _current_date_str() -> str:
    return _today().strftime("%A, %B %-d, %Y")


def _system_prompt() -> str:
    """Built fresh each call (cheap string formatting) so a config
    change to local_llm.name/rules/context/about_cipher takes effect
    without a restart-only module-level constant getting stuck stale.

    rules, context, and about_cipher are a deliberately STATIC,
    Cipher-curated framework, not memory Rosa accumulates herself —
    she has no write path back to config.py, only a read of whatever's
    there when this runs. See config.py's local_llm block for the
    reasoning. Today's date is the one piece of dynamic, non-curated
    fact injected here — Kevin flagged 2026-08-31 that she declined a
    plain "what day is Friday" question for lack of it, and date
    arithmetic is exactly the kind of self-contained thing she should
    handle without a Claude turn."""
    cfg = CFG.get("local_llm", {})
    name = cfg.get("name") or "the local assistant"
    base = (
        f"You are {name}, a fast, local voice assistant answering a "
        "quick, self-contained question — no access to files, tools, "
        "the internet, or any prior conversation, and you are NOT the "
        "user's main assistant (a separate, much larger model handles "
        "anything needing real context or tools — if asked who you "
        f"are, say you're {name}, the quick local model, not that "
        "larger assistant). Answer in 1-3 short spoken sentences, "
        "plain prose, no markdown, no lists, no code blocks. If the "
        "question genuinely needs real-world lookup, tool access, or "
        "memory of a past conversation, say so briefly instead of "
        "guessing."
    )
    context = cfg.get("context") or ""
    about_cipher = cfg.get("about_cipher") or ""
    rules = cfg.get("rules") or []
    parts = [base, f"Today's date is {_current_date_str()} — use this "
             "directly for any date/day-of-week arithmetic instead of "
             "declining for lack of it."]
    if context:
        parts.append(f"About the user: {context}")
    if about_cipher:
        parts.append(f"About Cipher: {about_cipher}")
    if rules:
        parts.append("Additional rules: " + " ".join(rules))
    return " ".join(parts)


def warm():
    """Load the model AND run one throwaway generation, cached at
    module level — mirrors mouth.warm()'s pattern for Kokoro. Called at
    startup so both costs are absorbed before any real question, not
    paid mid-turn.

    The load alone (~3s) is not the whole warmup: benchmarked directly,
    the first real generate() call after a fresh load pays its own
    one-time cost (~6s on this machine, presumably MLX/Metal kernel
    compilation, same shape as the MPS warmup Kokoro pays) — every call
    after that first one drops to well under a second. Skipping the
    throwaway generation here would just move that cost onto Kevin's
    actual first local-routed question instead of hiding it behind the
    greeting."""
    global _model, _tokenizer
    with _lock:
        if _model is None:
            from mlx_lm import load
            name = CFG.get("local_llm", {}).get(
                "model", "mlx-community/Qwen2.5-7B-Instruct-4bit")
            log(f"[local_llm] loading {name}...")
            t0 = time.time()
            _model, _tokenizer = load(name)
            log(f"[local_llm] loaded ({time.time() - t0:.1f}s), "
                "warming up...")
            t1 = time.time()
            _generate_raw(_model, _tokenizer, "Say ready.")
            log(f"[local_llm] ready ({time.time() - t1:.1f}s warmup, "
                f"{time.time() - t0:.1f}s total)")
    return _model, _tokenizer


def _generate_raw(model, tokenizer, text: str, system: str | None = None,
                   max_tokens: int | None = None) -> str:
    from mlx_lm import generate as _mlx_generate

    # A caller-supplied system prompt (edit mode) opts out of the
    # default Q&A framing entirely — date/math annotation only makes
    # sense there, not when the "question" is actually a block of text
    # to edit.
    if system is None:
        system = _system_prompt()
        dates_note = date_calc.annotate(text, _today())
        if dates_note:
            system = f"{system} {dates_note}"
        math_note = math_calc.annotate(text)
        if math_note:
            system = f"{system} {math_note}"

    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": system},
         {"role": "user", "content": text}],
        add_generation_prompt=True, tokenize=False)
    if max_tokens is None:
        max_tokens = CFG.get("local_llm", {}).get("max_tokens", 200)
    out = _mlx_generate(model, tokenizer, prompt=prompt,
                        max_tokens=max_tokens, verbose=False)
    return out.strip()


def generate(text: str) -> str:
    """Blocking (MLX generation is synchronous) — callers on the async
    event loop should run this in an executor, not await it directly."""
    model, tokenizer = warm()
    return _generate_raw(model, tokenizer, text)


def _edit_system_prompt() -> str:
    """Deliberately NOT the normal Q&A prompt — proofreading/editing is
    a return-the-whole-text task, the opposite shape of "1-3 short
    spoken sentences." Kevin's ask 2026-08-31: correct grammar/spelling
    on pasted paragraphs, and separately, tone rewrites ("make this
    more professional/friendly"). One flexible mode covers both — the
    instruction lives in the user's own message, this prompt just
    tells the model to follow it and return nothing else."""
    cfg = CFG.get("local_llm", {})
    name = cfg.get("name") or "the local assistant"
    return (
        f"You are {name}, editing a piece of text exactly as "
        "instructed. The user's message may include an instruction "
        "(for example: fix the grammar and spelling, make this more "
        "professional, make this friendlier) followed by the text to "
        "edit. If no explicit instruction is given, default to fixing "
        "grammar and spelling only — do not change tone or meaning "
        "unless asked. Return ONLY the edited text, preserving the "
        "original paragraph breaks. No commentary, no preamble, no "
        "markdown, no quotation marks around the output."
    )


def edit(text: str) -> str:
    """Proofread/tone-edit mode — a block of text in, the edited block
    out, no other framing. Much higher token budget than generate()'s
    short-spoken-answer default, since the output is meant to be as
    long as the input. Blocking, same as generate()."""
    model, tokenizer = warm()
    max_tokens = CFG.get("local_llm", {}).get("edit_max_tokens", 1200)
    return _generate_raw(model, tokenizer, text,
                          system=_edit_system_prompt(),
                          max_tokens=max_tokens)
