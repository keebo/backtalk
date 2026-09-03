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
"""The mouth — streaming sentence-chunked TTS, played through one
long-lived output stream.

Default engine: Kokoro, in-process. Local, free, no server, no API key,
~0.2s to first audio once warm. Optional premium engine: ElevenLabs on
YOUR key — read from the system keychain, never from a file (see
_get_elevenlabs_key) — with Kokoro as the automatic fallback: the voice
degrades instead of going mute if the cloud fails.

Sentences are synthesized one at a time and queued for playback, so the
first sentence is audible while later ones are still rendering. Playback
is cancellable mid-word: set the stop event and the speaker goes silent
within one audio block plus the device buffer (~0.15s).

HARD-WON AUDIO LAW #1 — ONE long-lived OutputStream, reused for every
sentence for the life of the process. A fresh stream per sentence gives
an audible onset blip or a beat of dead air on plenty of audio setups
(USB interfaces, Bluetooth, streaming mixers that latch onto each new
stream late). Proven by A/B test; do not "simplify" this away.

HARD-WON AUDIO LAW #2 — buffer ~0.75s of synthesized audio before a
sentence starts playing, so a slower machine never underruns into
slow-motion garble.
"""
import os
import queue
import re
import shutil
import sys
import tempfile
import threading

import numpy as np
import sounddevice as sd

from backtalk.config import CFG
from backtalk.vlog import log, log_debug

KOKORO_RATE = 24000
EL_RATE = 44100
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# A device that's gone silently stale (e.g. after the system sat idle) can
# keep reporting underflow on every write forever without ever raising an
# exception — so _drop_out()'s existing exception-triggered self-heal never
# fires. This is the backstop: a run of consecutive underflow hits within
# one sentence forces the same recovery exceptional-path takes. This does
# knowingly cost the audio-law-#1 stream continuity (a brief onset blip) —
# accepted here because the alternative, confirmed live 2026-09-02, is
# staying silent for the rest of the session.
_UNDERFLOW_RECOVERY_THRESHOLD = 10

_pipe = None
_pipe_lock = threading.Lock()

_mlx_model = None
_mlx_lock = threading.Lock()


def _ensure_espeak():
    """kokoro phonemizes through system espeak-ng (its bundled loader
    ships a broken build path — found the hard way; upstream's own docs
    say install the system package). Help phonemizer find it in the
    usual homes when the env isn't already set."""
    # Cosmetic, but Kevin kept seeing it: misaki's espeak backend logs a
    # "words count mismatch" warning through Python's standard logging
    # whenever espeak merges/drops a word during phonemization (normal,
    # harmless — confirmed no audible correlation). Setting the logger's
    # level doesn't stick: phonemizer.logger.get_logger() unconditionally
    # resets both the level AND the handlers back to its own defaults
    # every time it's called internally, clobbering an external setLevel()
    # (verified directly — reproduced the exact warning, tried setLevel()
    # first, it still printed). Patching the INSTANCE's .warning method
    # survives that, since get_logger() mutates the existing singleton
    # logger's state but never touches already-bound instance attributes.
    import logging
    logging.getLogger("phonemizer").warning = lambda *a, **k: None
    if os.environ.get("PHONEMIZER_ESPEAK_LIBRARY"):
        return
    candidates = (
        "/opt/homebrew/lib/libespeak-ng.dylib",       # macOS arm64 (brew)
        "/usr/local/lib/libespeak-ng.dylib",          # macOS intel (brew)
        "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1",  # debian/ubuntu
        "/usr/lib/libespeak-ng.so.1",                 # other linux
        "C:\\Program Files\\eSpeak NG\\libespeak-ng.dll",       # windows
        "C:\\Program Files (x86)\\eSpeak NG\\libespeak-ng.dll",
    )
    for lib in candidates:
        if os.path.exists(lib):
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = lib
            break


# Every espeak library filename phonemizer might copy, on any platform. A
# directory holding exactly one of these and nothing else is a phonemizer
# scratch dir and is not plausibly anything else.
_ESPEAK_LIB_NAMES = (
    "espeak-ng.dll",
    "libespeak-ng.dll",
    "libespeak-ng.so",
    "libespeak-ng.so.1",
    "libespeak-ng.dylib",
)


def _is_orphan_espeak_tempdir(path: str) -> bool:
    """True only for a directory whose ENTIRE contents are one espeak
    library. That signature is what makes it safe to point a delete at a
    shared temp folder: one file, and its name is one of five."""
    try:
        entries = os.listdir(path)
    except OSError:
        return False
    return len(entries) == 1 and entries[0] in _ESPEAK_LIB_NAMES


def _sweep_orphan_espeak_tempdirs():
    """Delete espeak scratch dirs left behind by previous runs.

    phonemizer copies the espeak shared library into a fresh temp dir for
    every backend it builds, because espeak-ng keeps its state in globals
    and the loader refuses the same file twice. Kokoro builds several
    backends, so ONE start leaves several behind.

    On POSIX that cleanup rides a finalizer and usually happens. On
    Windows phonemizer can only register it with atexit, and atexit does
    not run when a process is KILLED rather than exited -- so anything
    stopping the voice line by terminating it, which is most launchers and
    every supervisor, leaks every directory it ever made. Sixty had piled
    up on the machine where this was found, and fifteen were sitting on
    the author's own Mac when it was reviewed: the POSIX path is not as
    reliable as it looks either. The count only ever grows.

    Patching phonemizer where it is installed is not a fix, because the
    launcher runs a dependency sync that would overwrite it. Sweeping at
    our own startup bounds the total at one run's worth instead.

    Two things make deleting from a shared temp folder safe, and only the
    first is ours: the signature above is narrow enough that nothing else
    matches it, and anything we are not permitted to remove raises and is
    skipped. On Windows a loaded library cannot be deleted at all, so a
    live instance is protected by the OS rather than by us noticing it.
    POSIX does not work that way, but a process that has already mapped
    the library keeps it after the unlink, so a running instance is
    unharmed either way.
    """
    root = tempfile.gettempdir()
    swept = 0
    try:
        names = os.listdir(root)
    except OSError:
        return
    for name in names:
        path = os.path.join(root, name)
        if not os.path.isdir(path) or not _is_orphan_espeak_tempdir(path):
            continue
        try:
            shutil.rmtree(path)
            swept += 1
        except OSError:
            pass          # in use, or not ours. Leaving it is correct.
    if swept:
        log(f"[mouth] swept {swept} orphaned espeak temp dir(s)")


_hf_token_cache: str | None = None


def _get_hf_token() -> str:
    """HF Hub token from macOS Keychain (item `backtalk-hf-token`), so
    Kokoro's model download isn't rate-limited as an anonymous request.
    Seed it once with:
      security add-generic-password -a "$USER" -s backtalk-hf-token -T /usr/bin/security -w
    Falls back to the HF_TOKEN environment variable if Keychain has no entry."""
    global _hf_token_cache
    if _hf_token_cache is not None:
        return _hf_token_cache
    import subprocess
    token = ""
    try:
        if sys.platform == "darwin":
            r = subprocess.run(["security", "find-generic-password",
                                "-s", "backtalk-hf-token", "-w"],
                               capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                token = r.stdout.strip()
    except Exception:
        pass
    _hf_token_cache = token or os.environ.get("HF_TOKEN", "")
    return _hf_token_cache


def warm():
    """Load the Kokoro pipeline (first call downloads the model to the
    HF cache). Called at startup while the greeting text is composed."""
    global _pipe
    with _pipe_lock:
        if _pipe is None:
            _ensure_espeak()
            if token := _get_hf_token():
                os.environ["HF_TOKEN"] = token
            # Before kokoro makes this run's scratch dirs, clear the ones
            # earlier runs could not clean up on their way out.
            _sweep_orphan_espeak_tempdirs()
            from kokoro import KPipeline
            # The voice name's first letter IS the language pipeline:
            # a=American English, b=British English, e/f/h/i/j/p/z = the
            # other shipped languages. bm_lewis -> 'b'.
            lang = (CFG["voice"] or "bm_lewis")[0]
            # MPS is ~2.4x faster than CPU here in raw steady-state
            # throughput (benchmarked 2026-08-29), but Kokoro's vocoder
            # doesn't use a real FFT for its STFT/iSTFT — it uses a
            # conv1d/conv_transpose1d approximation (kokoro/custom_stft.py,
            # built for ONNX-export compatibility, avoiding complex-number
            # ops), and PyTorch's MPS conv kernels compute that
            # approximation with genuinely different numerics than CPU's.
            # Confirmed live 2026-08-31: Kevin heard MPS output as
            # consistently less clear than CPU's, matching an earlier
            # finding that MPS wasn't corrupting the signal (no clipping,
            # no NaN) but WAS computing a meaningfully different
            # rendering (large residual vs. CPU, low-frequency-dominated).
            # So "auto" now means CPU by default — a real, audible
            # accuracy regression outweighs the raw speed gain for a
            # voice assistant. voice_device still lets this be overridden
            # ("mps" or "cpu") from config without touching this code.
            override = CFG.get("voice_device", "auto")
            device = None
            if override == "mps":
                device = "mps"
            elif override != "cpu" and override != "auto" \
                    and sys.platform == "darwin":
                # Unrecognized value — fall back to the old MPS-preferring
                # auto-detect rather than silently treating a typo as "cpu".
                import torch
                if torch.backends.mps.is_available():
                    device = "mps"
            log(f"[mouth] loading kokoro (lang '{lang}', "
                f"voice {CFG['voice']}, device {device or 'auto'})...")
            _pipe = KPipeline(lang_code=lang, device=device)
            log("[mouth] voice ready")
    return _pipe


def warm_mlx():
    """Load Kokoro through mlx-audio (Apple's own MLX framework, not
    PyTorch/MPS) — the alternative voice_backend, added 2026-08-31.

    Confirmed live: sidesteps the PyTorch-MPS conv-kernel accuracy bug
    documented in warm()'s own comment above (MPS computing a genuinely
    different, less clear rendering of Kokoro's conv-based STFT
    approximation) — Kevin heard this backend as at least as clear as
    CPU, while a standalone benchmark measured it dramatically faster
    in steady state than either the old CPU or MPS/PyTorch paths (real
    time factor ~0.04-0.10 once warm, vs. PyTorch's best case of
    beating real-time by only ~2.4x on MPS). See
    08 - Resources/MLX-Audio Kokoro Prototype.md in the vault for the
    full standalone verification this was built on before touching
    this file."""
    global _mlx_model
    with _mlx_lock:
        if _mlx_model is None:
            _ensure_espeak()
            from mlx_audio.tts.utils import load_model
            name = CFG.get("voice_backend_model") \
                or "mlx-community/Kokoro-82M-bf16"
            log(f"[mouth] loading kokoro via mlx-audio ({name})...")
            _mlx_model = load_model(name)
            log("[mouth] voice ready (mlx)")
    return _mlx_model


def _stream_kokoro_mlx(text: str, voice: str | None = None):
    """One sentence -> int16 PCM chunks at 24kHz, via mlx-audio. Same
    output contract as _stream_kokoro (same sample rate, same dtype),
    so synth_stream()'s caller needs no changes either way."""
    model = warm_mlx()
    v = voice or CFG["voice"]
    lang = (v or "bm_lewis")[0]
    try:
        speed = float(CFG.get("speed") or 1.0)
    except (TypeError, ValueError):
        speed = 1.0
    for result in model.generate(text, voice=v, lang_code=lang, speed=speed):
        a = np.asarray(result.audio, dtype=np.float32)
        if a.size:
            yield (np.clip(a, -1.0, 1.0) * 32767).astype(np.int16)


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_RE.split(text.strip()) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _stream_kokoro(text: str, voice: str | None = None):
    """One sentence -> int16 PCM chunks at 24kHz, in-process. `voice`
    overrides CFG["voice"] for this call only — used to give the local
    LLM's replies an audibly different voice from Cipher's own."""
    pipe = warm()
    try:
        speed = float(CFG.get("speed") or 1.0)
    except (TypeError, ValueError):
        speed = 1.0
    for _, _, audio in pipe(text, voice=voice or CFG["voice"], speed=speed):
        a = np.asarray(audio, dtype=np.float32)
        if a.size:
            yield (np.clip(a, -1.0, 1.0) * 32767).astype(np.int16)


def _stream_elevenlabs(text: str, timeout: float):
    """ElevenLabs -> ffmpeg streaming decode -> int16 PCM at 44.1kHz.

    THE ELEVENLABS DOCTRINE, learned the expensive way:
    - fetch mp3_44100_128 and decode locally (raw 44.1k PCM needs their
      Pro tier; the mp3 decode hides inside network wait anyway)
    - turbo model, stability 0.5, similarity 0.75
    - never the multilingual model for English, never style above 0 —
      both make delivery slow and dull
    - their site previews are MASTERED demo clips; raw API output never
      matches them, so master locally (the ffmpeg chain in config)
    ffmpeg reads stdin as we feed it, so playback still starts before
    synthesis finishes."""
    import subprocess

    import httpx

    el = CFG["elevenlabs"]
    key = _get_elevenlabs_key()
    url = (f"https://api.elevenlabs.io/v1/text-to-speech/"
           f"{el['voice_id']}/stream?output_format=mp3_44100_128")
    proc = subprocess.Popen(
        ["ffmpeg", "-loglevel", "quiet", "-i", "pipe:0",
         "-af", el["master"],
         "-f", "s16le", "-ar", str(EL_RATE), "-ac", "1", "pipe:1"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    feed_error: list = []

    def _feed():
        try:
            with httpx.stream("POST", url, headers={"xi-api-key": key},
                              json={"text": text, "model_id": el["model"],
                                    "voice_settings": {
                                        "stability": 0.5,
                                        "similarity_boost": 0.75}},
                              timeout=timeout) as r:
                r.raise_for_status()
                for chunk in r.iter_bytes(chunk_size=4096):
                    proc.stdin.write(chunk)
        except Exception as e:
            feed_error.append(e)
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass

    t = threading.Thread(target=_feed, daemon=True)
    t.start()
    carry = b""
    got_audio = False
    while True:
        data = proc.stdout.read(8820)
        if not data:
            break
        data = carry + data
        usable = len(data) - (len(data) % 2)
        carry = data[usable:]
        if usable:
            got_audio = True
            yield np.frombuffer(data[:usable], dtype=np.int16)
    proc.wait(timeout=10)
    if feed_error and not got_audio:
        raise feed_error[0]


_el_key_cache: str | None = None


def _key_slot() -> str:
    """The credential-store entry name, so someone who already keeps a key
    under their own name points at it instead of storing a second copy."""
    return str(CFG["elevenlabs"].get("key_slot") or "backtalk-elevenlabs")


def _get_elevenlabs_key() -> str:
    """The API key, from the most secure store available — NEVER from a
    file in this repo. Lookup order:
      1. macOS Keychain, item `backtalk-elevenlabs` by default (change it
         with elevenlabs.key_slot) — seed it once with:
         security add-generic-password -a "$USER" -s backtalk-elevenlabs -T /usr/bin/security -w
         (it prompts for the secret; -T lets this code read it without a
         GUI prompt every launch)
      2. Linux secret-tool (libsecret):
         secret-tool store --label backtalk service backtalk-elevenlabs
      3. the ELEVENLABS_API_KEY environment variable — the last-resort
         fallback, and the only option on Windows for now. Know the
         tradeoff: an export line in a shell profile is a plaintext key
         on disk, which is exactly what the keychain path avoids."""
    global _el_key_cache
    if _el_key_cache is not None:
        return _el_key_cache
    import subprocess
    key = ""
    try:
        if sys.platform == "darwin":
            r = subprocess.run(["security", "find-generic-password",
                                "-s", _key_slot(), "-w"],
                               capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                key = r.stdout.strip()
        elif sys.platform.startswith("linux"):
            from shutil import which
            if which("secret-tool"):
                r = subprocess.run(["secret-tool", "lookup", "service",
                                    _key_slot()],
                                   capture_output=True, text=True, timeout=5)
                if r.returncode == 0:
                    key = r.stdout.strip()
    except Exception:
        pass
    _el_key_cache = key or os.environ.get("ELEVENLABS_API_KEY", "")
    return _el_key_cache


def _elevenlabs_ready() -> bool:
    el = CFG["elevenlabs"]
    return bool(el.get("enabled") and el.get("voice_id")
                and _get_elevenlabs_key())


def synth_stream(text: str, timeout: float = 30.0, voice: str | None = None):
    """One sentence -> yields (sample_rate, pcm_chunk) as the TTS
    renders. ElevenLabs when configured, Kokoro otherwise — and Kokoro
    as the fallback on ANY ElevenLabs failure. Degrade, never mute.

    `voice` forces a specific Kokoro voice for this call and skips
    ElevenLabs entirely — an override always means "speak this in a
    different, distinguishable Kokoro voice," which ElevenLabs can't do
    with a Kokoro voice name."""
    if voice is None and _elevenlabs_ready():
        try:
            for pcm in _stream_elevenlabs(text, timeout):
                yield EL_RATE, pcm
            return
        except Exception as e:
            log(f"[mouth] elevenlabs failed ({str(e)[:60]}) — "
                f"falling back to {CFG['voice']}")
    stream_fn = (_stream_kokoro_mlx
                 if CFG.get("voice_backend") == "mlx" else _stream_kokoro)
    for pcm in stream_fn(text, voice=voice):
        yield KOKORO_RATE, pcm


def _default_output_name() -> str | None:
    """Name of the current default output device, or None if it can't be
    read. Used to catch a system-level output switch (e.g. dock -> Mac
    speakers) that happens while our stream sits open — PortAudio won't
    raise on that by itself, so nothing else would notice."""
    try:
        idx = sd.default.device[1]
        return sd.query_devices()[idx]["name"]
    except Exception:
        return None


class Mouth:
    def __init__(self):
        from backtalk.ducking import Ducker
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._speaking = threading.Event()
        # The one persistent output stream (audio law #1).
        # Worker-thread-only — never touch from other threads.
        self._out: sd.OutputStream | None = None
        self._out_rate: int | None = None
        self._out_device: str | None = None
        self._consecutive_underflows = 0
        self.ducker = Ducker()  # public: PTT ducks for the USER's voice too
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()
        # Late-bound (main.py sets this once the brain exists): tells the
        # worker whether a turn is still in flight, so draining the local
        # speech queue mid-turn doesn't get mistaken for the reply being over.
        self._turn_active = None
        # Late-bound alongside _turn_active: what the bus should show if
        # the turn's still going once this sentence is done playing. Without
        # this the "speaking" write below just sits stale through however
        # much of the tool call is left — see the finally block.
        self._turn_state = None

    @property
    def speaking(self) -> bool:
        return self._speaking.is_set()

    def nothing_queued(self) -> bool:
        """Nothing left to play right now. Used at turn-end to settle the
        face when a tool call was the very last thing (no trailing text
        ever arrived to trigger the worker's own idle check)."""
        return self._q.empty() and not self._speaking.is_set()

    def say(self, text: str, voice: str | None = None):
        """Queue text (split to sentences) for speech. `voice` overrides
        the configured voice for every sentence in this call — used for
        the local LLM's replies."""
        for s in split_sentences(text):
            self._q.put((s, None, voice))

    def say_chunk(self, text: str, directions=None, voice: str | None = None):
        """Queue text as ONE TTS request, no sentence splitting — fuller
        chunks get livelier prosody (single short sentences come out
        dull).

        `directions` are the stage directions this chunk carried. They are
        published on the signal bus when this chunk's audio STARTS, which
        is why they travel with it instead of firing at parse time."""
        text = text.strip()
        if text:
            self._q.put((text, directions or None, voice))

    def shut_up(self):
        """Barge-in: stop current playback and flush everything queued."""
        self._stop.set()
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass

    def shutdown(self):
        """Exit path: stop playback and restore the music SYNCHRONOUSLY
        (the debounced restore timer dies with the process otherwise)."""
        self.shut_up()
        self.ducker.restore_now()

    def wait_done(self, timeout: float | None = None):
        """Block until the queue is drained and playback finished."""
        import time
        deadline = None if timeout is None else time.time() + timeout
        while (not self._q.empty()) or self._speaking.is_set():
            time.sleep(0.05)
            if deadline and time.time() > deadline:
                return

    def _run(self):
        from backtalk import signals
        while True:
            item = self._q.get()
            sentence, directions, voice = item
            if not sentence:
                continue
            self._stop.clear()
            self._speaking.set()
            self.ducker.speech_start()
            signals.static_stop()     # thinking sound dies when speech starts
            signals.set_state("speaking")
            try:
                self._play_stream(sentence, directions, voice=voice)
            except Exception as e:
                log(f"[mouth] synth/play error: {e}")
            finally:
                if self._q.empty():
                    self._speaking.clear()
                    # The reply has genuinely stopped talking, as opposed to
                    # the gap between two sentences of the same reply.
                    signals.reply_done()
                    self.ducker.speech_end()
                    # Only declare idle if the brain agrees the turn is
                    # actually over. Mid-turn (a tool call about to run,
                    # or more text still coming) this queue drains too —
                    # restore whatever brain.ask_stream currently wants
                    # shown instead of leaving the bus on "speaking",
                    # which this same loop just wrote and which is stale
                    # the moment audio actually stops.
                    if self._turn_active is None or not self._turn_active():
                        signals.set_state("idle")
                    elif self._turn_state is not None:
                        signals.set_state(self._turn_state())

    def _get_out(self, rate: int) -> sd.OutputStream:
        """The long-lived stream (audio law #1). Reopened when the sample
        rate changes (ElevenLabs 44.1k <-> Kokoro 24k fallback: rare,
        costs at most one blip on the switch), or when the default output
        device itself has changed since we opened it (e.g. the user
        switches from headphones to speakers) — PortAudio doesn't error
        on that, it just keeps writing into a stream that no longer
        matches reality, which is how you get static instead of silence."""
        current_device = _default_output_name()
        device_changed = (current_device is not None
                           and self._out_device is not None
                           and current_device != self._out_device)
        if self._out is not None and self._out_rate == rate and not device_changed:
            # Guarded, because the stream can die UNDER us: the ears
            # rebuild the whole audio system to recover from a device
            # change (see ears._reopen_after_device_change), and that
            # closes every open stream including this one. Touching a
            # dead stream raises rather than returning False, so the
            # check has to be the try, not an `if`. Falling through
            # rebuilds it, which is what the rest of this method does.
            try:
                if not self._out.active:
                    self._out.start()
                return self._out
            except Exception:
                log("[mouth] the output stream went away, reopening")
        if device_changed:
            log(f"[mouth] default output changed ({self._out_device!r} -> "
                f"{current_device!r}), reopening")
        self._drop_out()
        self._out = sd.OutputStream(samplerate=rate, channels=1, dtype="int16")
        self._out_rate = rate
        self._out_device = current_device
        self._out.start()
        return self._out

    def _cut(self):
        """Barge-in cut: stop feeding audio and pad the line with a beat
        of silence — the stream itself NEVER stops (an abort+restart here
        re-triggers the onset blip on latch-happy audio setups). Cost:
        the device buffer (~0.1s) plays out after the kill order — half a
        syllable of tail."""
        try:
            zeros = np.zeros(2205, dtype=np.int16)
            for _ in range(3):
                self._out.write(zeros)
        except Exception:
            self._drop_out()

    def _drop_out(self):
        """Close and forget the stream — the next sentence reopens
        fresh. The self-heal path for device errors (interface
        unplugged, audio mixer restarted)."""
        if self._out is not None:
            try:
                self._out.close(ignore_errors=True)
            except Exception:
                pass
        self._out = None
        self._out_rate = None

    def _play_stream(self, sentence: str, directions=None, block: int = 2205,
                     prebuffer_s: float = 0.75, voice: str | None = None):
        """Stream-synthesize and play with the head-start buffer (audio
        law #2). stop() reacts ~50ms. The sample rate comes from
        whichever engine actually answered."""
        from backtalk import signals
        gen = synth_stream(sentence, voice=voice)
        head: list = []
        banked = 0
        rate = None
        for rate_, pcm in gen:
            rate = rate_
            head.append(pcm)
            banked += len(pcm)
            if banked >= int(rate * prebuffer_s):
                break
        if rate is None:
            return
        try:
            out = self._get_out(rate)
            # AUDIO STARTS HERE: the head buffer is full and the first write
            # is next. Publishing now is what puts a screen cue on the spoken
            # word rather than seconds ahead of it.
            if directions:
                from backtalk import signals as _sig
                _sig.direction(directions)

            def _write(pcm):
                nonlocal out
                for i in range(0, len(pcm), block):
                    if self._stop.is_set():
                        return False
                    # write() returns True if PortAudio detected the
                    # output buffer starve before this call. Confirmed
                    # 2026-09-01 this fires far more often than it's
                    # actually audible — likely this loop's own
                    # feed_waveform() call below doing a disk write on
                    # nearly every chunk (its ~67ms throttle is shorter
                    # than a chunk's own ~92ms playback time), so a slow
                    # write can starve the buffer for a few harmless ms.
                    # Kept as log_debug (file only, not the live
                    # transcript) rather than removed entirely: sparse
                    # hits are noise, but a dense SUSTAINED run of these
                    # is still the real signature a serious recurrence
                    # (e.g. the Boom3D driver issue) would leave behind.
                    if out.write(pcm[i:i + block]):
                        self._consecutive_underflows += 1
                        log_debug("[mouth] output underflow — audio buffer starved")
                        if self._consecutive_underflows >= _UNDERFLOW_RECOVERY_THRESHOLD:
                            # The sustained-run signature the comment above
                            # warns about: the device is stale, not just
                            # momentarily slow. write() will keep silently
                            # "succeeding" forever on a device like this, so
                            # nothing else self-heals it — force a fresh
                            # stream now rather than staying silent for the
                            # rest of the session.
                            log("[mouth] sustained underflow — reopening output stream")
                            self._drop_out()
                            out = self._get_out(rate)
                            self._consecutive_underflows = 0
                    else:
                        self._consecutive_underflows = 0
                    # Re-check after the blocking write: a barge-in
                    # landing mid-block must not let feed_waveform
                    # re-assert "speaking" over a fresh "listening".
                    if self._stop.is_set():
                        return False
                    signals.feed_waveform(pcm[i:i + block])
                return True
            for pcm in head:
                if not _write(pcm):
                    self._cut()
                    return
            for _, pcm in gen:
                if not _write(pcm):
                    self._cut()
                    return
        except Exception:
            self._drop_out()
            raise


if __name__ == "__main__":
    m = Mouth()
    m.say(sys.argv[1] if len(sys.argv) > 1 else
          "Voice check. The mouth is alive, and it is very good to be heard.")
    m.wait_done(timeout=60)
