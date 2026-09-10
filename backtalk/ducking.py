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
"""Spotify ducking (macOS) — music dips while the voice talks.

Relative duck: if Spotify is playing above THRESHOLD, drop it to
max(THRESHOLD, current * PCT) while the assistant speaks; restore after.
Quiet music is left alone. The restore is DEBOUNCED: the mouth goes
momentarily idle between streamed chunks, and bouncing the volume every
gap is seasickness — restore only after sustained silence. Never
launches Spotify; every call no-ops if it isn't running.

On non-macOS platforms every method is a silent no-op (the AppleScript
bridge is the macOS way; PRs for pycaw/playerctl equivalents welcome).
"""
import subprocess
import sys
import threading

THRESHOLD = 30
PCT = 0.60
RESTORE_DEBOUNCE_S = 0.5

_DARWIN = sys.platform == "darwin"

# Cached per process lifetime, not permanently -- a fresh backtalk restart
# re-checks, so installing Spotify later just works with no code change.
# Root-caused 2026-09-10: with Spotify never installed, `_osa`'s "is
# running" query below still had to ask macOS to resolve the app by name,
# which is enough to trigger a fresh Automation permission prompt every
# single speech_start() call -- forever, since no target ever existed to
# record a lasting Allow/Deny against. `mdfind` is a pure Spotlight
# metadata lookup, not an AppleEvent, so it never touches that permission
# gate at all -- checking installation this way, once, avoids ever asking
# macOS about Spotify in the first place when it's simply not there.
_spotify_installed_cache: bool | None = None


def _spotify_installed() -> bool:
    global _spotify_installed_cache
    if _spotify_installed_cache is None:
        if not _DARWIN:
            _spotify_installed_cache = False
        else:
            try:
                r = subprocess.run(
                    ["mdfind", "kMDItemCFBundleIdentifier == 'com.spotify.client'"],
                    capture_output=True, text=True, timeout=2.0,
                )
                _spotify_installed_cache = bool(r.stdout.strip())
            except Exception:
                _spotify_installed_cache = False
    return _spotify_installed_cache


def _osa(script: str, timeout: float = 2.0) -> str | None:
    if not _DARWIN:
        return None
    try:
        r = subprocess.run(["osascript", "-e", script],
                           capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception:
        return None


def _spotify_volume() -> int | None:
    if not _spotify_installed():
        return None
    if _osa('application "Spotify" is running') != "true":
        return None
    v = _osa('tell application "Spotify" to get sound volume')
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _set_volume(level: int):
    _osa(f'tell application "Spotify" to set sound volume to {int(level)}')


class Ducker:
    def __init__(self):
        self._lock = threading.Lock()
        self._original: int | None = None
        self._timer: threading.Timer | None = None

    def speech_start(self):
        """Duck (once) when speech starts; cancel any pending restore."""
        with self._lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None
            if self._original is not None:
                return  # already ducked
            current = _spotify_volume()
            if current is None or current <= THRESHOLD:
                return
            target = max(THRESHOLD, int(current * PCT))
            if target >= current:
                return
            self._original = current
            _set_volume(target)

    def speech_end(self, debounce: float = RESTORE_DEBOUNCE_S):
        """Schedule a debounced restore; resumed speech cancels it."""
        with self._lock:
            if self._original is None:
                return
            if self._timer:
                self._timer.cancel()
            self._timer = threading.Timer(debounce, self._restore)
            self._timer.daemon = True
            self._timer.start()

    def _restore(self):
        with self._lock:
            if self._original is not None:
                _set_volume(self._original)
                self._original = None
            self._timer = None

    def restore_now(self):
        """Synchronous restore for shutdown paths — the debounce timer is
        a daemon thread and dies with the process, which otherwise leaves
        the music stuck quiet after you hang up. Call before any exit."""
        with self._lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None
        self._restore()
