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
"""Configuration — backtalk.json in the repo root, merged over defaults.

backtalk deliberately owns NO personality. Your agent's identity lives in
the CLAUDE.md of whatever folder `agent_dir` points at — backtalk just
gives that agent a mouth and ears. The only voice-related instruction it
adds is the spoken-delivery discipline below, which is about the MEDIUM
(writing for the ear), never the character.
"""
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# One install, more than one assistant. Point BACKTALK_CONFIG at a different
# JSON file and you get a second agent (its own name, voice, folder and
# greeting) without a second copy of the code. A launcher exports it; nothing
# else changes.
CONFIG_PATH = Path(os.environ.get("BACKTALK_CONFIG") or (REPO / "backtalk.json"))

DEFAULTS = {
    # The folder whose CLAUDE.md defines WHO your agent is. The voice
    # session runs there, so it's the same assistant as your terminal
    # sessions — same name, same personality, same memory.
    "agent_dir": "~",
    # Display name, used in logs and to build the quit phrases
    # ("goodbye <name>" hangs up). Match your agent's actual name.
    "name": "Assistant",
    # The brain. Full model id ON PURPOSE — never a bare alias like
    # "sonnet": the SDK resolves aliases through its own bundled CLI and
    # can silently land on an older model. The fast tier is most of the
    # speed difference people ask about; a deep-work model makes every
    # reply noticeably slower and burns usage doing it.
    "model": "claude-sonnet-5",
    # The deep-work model for the voice console's "switch to the deep
    # model" command ("back to the fast model" returns to "model"
    # above). Full id ON PURPOSE, same reasoning as "model". The switch
    # lasts one session and is always spoken; this default never moves
    # by itself.
    "deep_model": "claude-opus-5",
    # Tool permissions for the voice session. "ask" is the default ON
    # PURPOSE (safety is opt-out, never opt-in): when the agent wants a
    # gated tool (write a file, run a real command), it ASKS OUT LOUD
    # and waits. Answer by voice or by typing. An EXACT yes approves
    # ("yes", "yeah", "go ahead", "approved"...); anything else denies,
    # and your words are passed back to the agent as the reason, so
    # "no, put it in drafts instead" actually steers it. No answer
    # within 75 seconds means no, out loud. Most read-only work passes
    # without asking; anything that changes things asks.
    # "bypassPermissions" is AUTO-APPROVE: the agent acts without
    # asking, exactly like a terminal session with approvals off.
    # (Not to be confused with hands-free LISTENING, which is about
    # the microphone: see mic_mode below.) Never hand-edit this file
    # to switch: tell your agent to change it (takes effect next
    # launch), or say "stop asking for permission" (then "confirm")
    # or "start asking again" inside a voice session for an immediate
    # flip that also saves. The legacy value "default" now
    # behaves as "ask" (a headless voice session could never render
    # the terminal prompt it promised).
    "permission_mode": "ask",
    # Which of your agent's skills the voice session can SEE. null keeps the
    # CLI's own default (all of them). [] hides every one. A list names the
    # ones to allow.
    #
    # This matters on a shared screen. Skill DESCRIPTIONS live in the system
    # prompt, so if yours name clients, employers or systems, they are one
    # screen-share away from an audience. A context filter, not a sandbox:
    # it decides what the session is TOLD about, not what it can reach.
    "visible_skills": None,
    # Extra folders the agent may access beyond agent_dir (e.g. your
    # notes vault). Absolute paths or ~ paths.
    "extra_dirs": [],
    # Hold-to-talk key. Named keys ("home", "f13", "right_alt", ...)
    # or a single character.
    "ptt_key": "home",
    # The microphone mode. "ptt" (push to talk, the default and the
    # recommendation): the mic is closed except while the key is held,
    # so room audio and your own speakers can never trigger the agent.
    # "open" (hands-free listening): always listening with voice
    # detection; a video, music with vocals, or another person in the
    # room CAN trigger it, and with open speakers it can hear itself
    # (headphones recommended). The key still works in hands-free
    # listening: it interrupts, and holding it always gets you heard.
    # Switch live by voice: "go hands free" / "push to talk mode"
    # (the switch saves itself here). The --open-mic launch flag
    # forces "open" for one session.
    "mic_mode": "ptt",
    # Playback speed for the built-in voice: 1.0 is Kokoro's native
    # pace, 1.15 is noticeably brisker, 0.9 is slower. Kokoro's own
    # pipeline implements it, so quality holds across sane values
    # (roughly 0.7 to 1.5). ElevenLabs pace lives in the master chain's
    # atempo instead. (Grew out of a community proposal, issue #1.)
    "speed": 1.0,
    # Resume the previous conversation on launch. OFF by default: a
    # fresh session every launch is the predictable behavior. Set true
    # and backtalk saves the session id after every completed turn
    # (signals_dir/.backtalk_session) and reattaches to it at the next
    # launch, so killing the window stops costing you the conversation.
    # A resume that fails falls back to a fresh session and says so in
    # the log. (Grew out of the same community proposal, issue #1.)
    "resume_last_session": False,
    # Publish your Claude usage (the five-hour and weekly windows) on the
    # signal bus so a face can draw it. OFF by default and deliberately
    # so: this is your own account spend, and the faces this feeds are
    # frequently on a stream or a shared screen. Nothing is collected at
    # all while this is false. (Community fix, ai-visualizer issue #1.)
    "show_usage": False,
    # Reasoning effort for the voice session: "" inherits the model's
    # default; "low" / "medium" / "high" / "max" applies at launch.
    # Saying "set effort to X" in a voice session saves itself here.
    "effort": "",
    # The voice (Kokoro, local, free). bm_lewis is the proven default —
    # British male, the butler register. Others: bm_george, bm_daniel,
    # bm_fable, am_michael, af_heart... The first letter picks the
    # language pipeline (a=American, b=British, e/f/h/i/j/p/z = other
    # languages), so keep voice and accent matched.
    "voice": "bm_lewis",
    # "auto" (CPU — see mouth.py's warm() for why), "cpu", or "mps".
    # Confirmed live 2026-08-31: Kokoro's MPS path is genuinely less
    # clear-sounding than CPU (a real numerics difference in its
    # conv-based STFT approximation, not just placebo), so "auto" means
    # CPU now despite MPS being ~2.4x faster in raw throughput — audio
    # correctness matters more than speed for a voice assistant. Set to
    # "mps" here to trade clarity back for speed if that tradeoff is
    # ever wanted again.
    "voice_device": "auto",
    # "mlx" (default — Kokoro via mlx-audio, Apple's own MLX framework)
    # or "pytorch" (the old KPipeline/MPS-or-CPU path, kept as a
    # fallback). MLX sidesteps the PyTorch-MPS conv-kernel accuracy bug
    # above entirely rather than working around it. Confirmed live
    # 2026-08-31, both voices Kevin actually uses (bm_lewis and Rosa's
    # bf_emma), through a real restarted conversation, not just a
    # standalone test — "sounds great" both times. Dramatically faster
    # in steady state than either PyTorch path too (see
    # 08 - Resources/MLX-Audio Kokoro Prototype.md).
    "voice_backend": "mlx",
    # Speech recognition (faster-whisper, local, free).
    # Models: tiny.en / base.en / small.en / medium.en — small.en is the
    # accuracy/speed sweet spot on a normal machine.
    "stt_model": "small.en",
    # "auto" uses CUDA when present, otherwise CPU. int8 keeps CPU fast.
    "stt_device": "auto",
    "stt_compute": "int8",
    # The microphone to record from, matched by NAME. "" means whatever
    # the OS calls the default input, which is right on most machines.
    #
    # Set a real device name to PIN the mic, so a headset connecting for
    # OUTPUT cannot steal your input -- which also keeps a Bluetooth
    # headset in high-quality A2DP instead of dropping it to the
    # narrowband call profile mid-sentence, degrading what you hear at
    # the same moment it takes your voice.
    #
    # A name and never an index: indices shift every time a device
    # connects or disconnects, the exact event this setting exists to
    # survive. Exact name wins, then the first case-insensitive
    # substring. A name matching nothing falls back to the default and
    # logs the inputs it did find; the mic degrades, it never goes mute.
    #
    # NOT "stt_device" below, which is the Whisper COMPUTE device.
    "mic_device": "",
    # Optional premium voice: ElevenLabs on YOUR key. The key NEVER
    # goes in a file: it's read from the macOS Keychain (item
    # `backtalk-elevenlabs`) or Linux secret-tool, with the
    # ELEVENLABS_API_KEY env var as last-resort fallback — see
    # mouth._get_elevenlabs_key for the seeding one-liners. Kokoro
    # remains the automatic fallback, so the voice degrades instead of
    # going mute if the cloud fails. Needs ffmpeg on the PATH.
    "elevenlabs": {
        "enabled": False,
        "voice_id": "",
        # Purely for you. Voice IDs are unreadable six months later, so put
        # the human name here; nothing reads it.
        "voice_note": "",
        "model": "eleven_turbo_v2_5",
        # Which OS credential-store entry holds the key. Change it if you
        # already keep an ElevenLabs key under a name of your own rather
        # than seeding a second copy of the same secret.
        "key_slot": "backtalk-elevenlabs",
        # Local mastering: ElevenLabs' site previews are mastered demo
        # clips and the raw API never matches them. This chain closes
        # the gap: presence lift, light chest, broadcast compression,
        # limiter. atempo is the one pace dial (1.0 = native).
        "master": ("atempo=1.12,highpass=f=70,"
                   "equalizer=f=3200:t=q:w=1.2:g=3.5,"
                   "equalizer=f=140:t=q:w=1:g=1.5,"
                   "acompressor=threshold=-18dB:ratio=2.5:attack=8:"
                   "release=120:makeup=4dB,alimiter=limit=0.95"),
    },
    # A local, on-device model (MLX) for self-contained questions that
    # don't need Claude's tools, vault access, or conversation memory —
    # a quick fact, some math, rephrasing a sentence. Every turn that
    # reaches Claude reloads the full system prompt and tool set
    # regardless of how simple the question is, so routing genuinely
    # self-contained stuff around that entirely saves real tokens. OFF
    # by default: needs `uv add mlx-lm` and a model download first, and
    # only makes sense on Apple Silicon. See router.py for how a turn
    # gets classified local vs cloud, and CLAUDE.md/docs for the voice
    # this speaks in (deliberately different from the main voice, so a
    # local answer is never mistaken for Cipher's own).
    "local_llm": {
        "enabled": False,
        "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "max_tokens": 200,
        # Proofread/tone-edit mode's own budget — the output is meant
        # to be as long as the input paragraph(s), unlike the short
        # spoken-answer default above. Kevin's ask, 2026-08-31.
        "edit_max_tokens": 1200,
        # Where proofread/edit mode writes its output — a plain-text
        # file Kevin can open and copy from, never spoken aloud in
        # full. Deterministic code writes this, not Rosa herself — she
        # has no file-write tool or agency here, same boundary as
        # always; see main.py's speak_reply_proofread.
        "edit_output_dir": "/Users/brown/Documents/Rosa",
        # A different Kokoro voice than "voice" above, same language
        # family so no second Kokoro pipeline has to load. Speaking in
        # a different voice IS the indicator that this answer came from
        # the small local model, not from Cipher.
        "voice": "bf_emma",
        # Its own name, deliberately not "Cipher"-anything: a wrong or
        # generic-sounding answer under Cipher's name reads as "Cipher
        # had a dumb moment" instead of "a much smaller model answered."
        # Used in log tags and told to the model itself in its own
        # system prompt (local_llm.py) so it can answer "who are you"
        # correctly. Also set ai-visualizer.json's local_name to match,
        # so the face shows this name instead of a generic "LOCAL" tag.
        "name": "Rosa",
        # A small, STATIC behavioral framework — Kevin's explicit ask,
        # 2026-08-31, after agreeing that open-ended/growing memory for
        # her would recreate the same reliability and governance gap
        # already ruled out for vault access. Both lists below are
        # meant to be curated deliberately (by Kevin or Cipher editing
        # this file), not something Rosa accumulates on her own —
        # she has no mechanism to write to this file, only read it.
        "rules": [
            "Speak conversationally and directly, not formally.",
            "Never claim to be Cipher/C.I.P.H.E.R. — you are a separate, "
            "much smaller local model.",
            "If a question needs specifics about Kevin's actual work, "
            "projects, files, or past conversations, say plainly that "
            "you don't have that and it should go to Cipher — never "
            "guess or invent specifics.",
        ],
        # Deliberately generic — no company names, project specifics, or
        # anything from a firewalled lane (P2P / Bravespan / Trading).
        # This block has no way to know which lane a given question
        # belongs to, so it stays lane-agnostic on purpose; anything
        # more specific belongs in a real turn with Cipher instead.
        "context": (
            "The user's name is Kevin. He's a working professional who "
            "values concise, direct, no-fluff answers over long "
            "explanations."
        ),
        # Who Cipher is, in relation to her — Kevin's explicit ask,
        # 2026-08-31, so she can answer accurately (not just deflect)
        # if asked to introduce herself or explain the relationship.
        "about_cipher": (
            "Cipher (full name C.I.P.H.E.R.) is Kevin's main AI "
            "assistant — the senior one in charge, with real tools, "
            "memory of past conversations, and access to Kevin's own "
            "files. You were built by Cipher specifically to answer "
            "small, self-contained questions locally so Kevin doesn't "
            "spend Claude usage on things that don't need it. If asked "
            "to introduce yourself, say plainly that you're Rosa, a "
            "fast local helper working alongside Cipher, who handles "
            "the real work."
        ),
    },
    # Where the signal-bus files are written (.voice_state,
    # .voice_waveform, .voice_loading_pid) — anything can watch them;
    # visualizers pair with this contract. Default: the repo root.
    "signals_dir": "",
    # THE BAREHANDS SEAM: point this at a barehands checkout's state/
    # folder and its on-screen ring becomes your agent's face — it
    # breathes while idle, spins while thinking, pulses with the voice.
    # (github.com/jaredrhod/barehands)
    "barehands_state_dir": "",
    # Sound played while the agent thinks, so a long pause never reads as
    # a dead line. The bundled one ships in assets/; a relative path
    # resolves against this repo. Set "" to think in silence.
    "thinking_sound": "assets/thinking.wav",
    # Spoken lines. {name} is replaced with "name" above.
    "greeting": "Voice line online. Hold {ptt_key} and talk to me.",
    # Spoken instead of "greeting" when mic_mode is "open", where telling
    # someone to hold a key is wrong. Leave "" to use "greeting" for both.
    "greeting_open_mic": "",
    "signoff": "Voice line closing. I'll be here when you need me.",
    # Appended to the spoken-delivery discipline below. The discipline covers
    # the MEDIUM (write for the ear, no markdown, keep it short); your agent's
    # CLAUDE.md covers the character. Use this for a note that belongs to
    # neither, e.g. a rule that only applies when it is speaking.
    "discipline_append": "",
}

# The spoken-delivery discipline — the MEDIUM half of what used to be a
# persona. The CHARACTER half deliberately is not here: it's whatever
# lives in the agent_dir's CLAUDE.md. One identity, one place.
DISCIPLINE = (
    "VOICE SESSION (your reply is spoken aloud through a TTS engine, "
    "not displayed): you are SPEAKING, in your own voice and "
    "personality — your CLAUDE.md is who you are. The TTS engine "
    "PERFORMS your punctuation, so write like a performance, never "
    "like a memo: contractions always, punchy conversational "
    "sentences, and if a line could open a quarterly report, rewrite "
    "it like you're telling a friend. Keep replies to a few short "
    "sentences; go longer only when the question genuinely needs it. "
    "No markdown, no lists, no code blocks, no emoji, no URLs. Say "
    "numbers the way a human says them out loud — never raw figures "
    "or symbols. NEVER SPEAK A FILE PATH: say the file, not its "
    "address. 'the config' or 'ears dot py', never a string of "
    "slashes and folder names read one by one — it is unbearable "
    "aloud and carries no meaning by ear. Same for URLs and long "
    "ids: name the thing, not the address. "
    "Skip any startup sequence; answer directly. "
    "VOICE CONSOLE FACTS, answer from these whenever the person asks "
    "you to change a voice-line setting: this session is controlled "
    "by exact spoken phrases, never by you. Permissions: 'stop "
    "asking for permission' (then 'confirm'), or 'start asking "
    "again'. Microphone: 'go hands free', or 'push to talk mode'. "
    "Also: 'clear the session', 'compact the session', 'switch to "
    "the deep model', 'back to the fast model', 'set effort to low' "
    "(or medium, high, max), and 'usage report'. You cannot flip "
    "these live yourself, so when asked, give the person the exact "
    "phrase to SAY. Editing backtalk.json only changes the default "
    "for the NEXT launch."
)


def _expand(p: str) -> str:
    return os.path.expanduser(p) if p else p


def load() -> dict:
    cfg = json.loads(json.dumps(DEFAULTS))          # deep copy
    try:
        user = json.loads(CONFIG_PATH.read_text())
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    except FileNotFoundError:
        pass
    except ValueError as e:
        print(f"[config] backtalk.json is not valid JSON ({e}) — "
              f"using defaults", flush=True)
    cfg["agent_dir"] = _expand(cfg["agent_dir"])
    cfg["extra_dirs"] = [_expand(d) for d in cfg.get("extra_dirs", [])]
    cfg["signals_dir"] = _expand(cfg.get("signals_dir", "")) or str(REPO)
    cfg["barehands_state_dir"] = _expand(cfg.get("barehands_state_dir", ""))
    thinking = _expand(cfg.get("thinking_sound", ""))
    if thinking and not os.path.isabs(thinking):
        thinking = str(REPO / thinking)
    cfg["thinking_sound"] = thinking
    name = str(cfg.get("name") or "Assistant")
    # Letters only: a display name like "C.I.P.H.E.R." must still build a
    # matchable quit phrase — speech-to-text never inserts the periods, so
    # matching on the punctuated name would never fire.
    low = "".join(ch for ch in name.lower() if ch.isalpha() or ch == " ")
    cfg["quit_phrases"] = tuple(cfg.get("quit_phrases") or (
        f"goodbye {low}", f"good bye {low}", "end voice mode",
        f"hang up {low}", "hang up"))
    key_label = "the " + str(cfg.get("ptt_key", "home")).replace("_", " ") \
                + " key"
    # In hands-free there is no key to hold, so a separate line can be set.
    if str(cfg.get("mic_mode", "ptt")) == "open" and cfg.get("greeting_open_mic"):
        cfg["greeting"] = cfg["greeting_open_mic"]
    cfg["greeting"] = str(cfg["greeting"]).replace(
        "{name}", name).replace("{ptt_key}", key_label)
    cfg["signoff"] = str(cfg["signoff"]).replace("{name}", name)
    return cfg


CFG = load()

# The character half stays in YOUR agent's CLAUDE.md. This is the medium.
if CFG.get("discipline_append"):
    DISCIPLINE = DISCIPLINE + " " + str(CFG["discipline_append"]).strip()
