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
"""Hold-to-talk — a global key listener.

HOLD the key -> mic opens. RELEASE -> mic closes and the utterance is
processed. The button IS the voice-activity detector, which is why this
mode is speaker-safe with no headphones: the mic simply isn't open while
the assistant talks, unless you press the key — and pressing while it
talks interrupts it.

THE KEY-REPEAT TRAP (the bug that kills every naive build): the OS fires
on_press events CONTINUOUSLY while a key is held. Without the held-state
filter below, every repeat reads as a fresh press and keeps cancelling
the reply before it can speak.

macOS needs Input Monitoring permission for the hosting terminal
(System Settings -> Privacy & Security -> Input Monitoring). Windows
works out of the box; some Linux desktops need the user in the `input`
group or an X11 session.

THE STUCK-HELD TRAP (macOS only): a screen lock/unlock or any focus
steal can suspend the input event tap mid-hold, and the matching
on_release for a modifier key (e.g. right_option) never arrives. Without
a check against reality, `_held` stays True forever — the mic looks
permanently "listening", and _on_press's key-repeat filter then refuses
to arm on the next real press, since it only arms when not already
held. A background watchdog polls the actual physical key state via
Quartz and resyncs `_held` if the event stream ever disagrees with it.

SELF-CONFIRMING GUARD (field-caught 2026-08-29): on at least one Mac,
`CGEventSourceKeyState` never once reports the configured key as down —
even while it's physically held — despite Terminal already holding both
Accessibility and Input Monitoring grants. Left ungated, the watchdog
then force-clears `_held` within one poll tick of every real press,
turning every genuine hold into a sub-0.25s "tap" that gets ignored:
push-to-talk looked completely dead. The watchdog now only earns the
right to CLEAR a stuck `_held` after it has itself witnessed the key
read as down at least once — proof the read actually works on this
machine. Until then it's inert, which just restores the old (working)
behavior instead of actively breaking every hold.
"""
import platform
import threading
import time

from pynput import keyboard

_HAS_QUARTZ = False
if platform.system() == "Darwin":
    try:
        from Quartz import (CGEventSourceKeyState,
                            kCGEventSourceStateHIDSystemState)
        _HAS_QUARTZ = True
    except ImportError:
        pass


def resolve_key(name: str):
    """'home' / 'f13' / 'right_alt' / any single character -> pynput key."""
    name = (name or "home").strip().lower()
    if len(name) == 1:
        return keyboard.KeyCode.from_char(name)
    # Friendly names -> pynput's names. pynput calls the right option key
    # alt_r, not right_alt; the docs speak human, this map translates.
    # (Field-caught: right_alt silently fell back to home, which Mac
    # laptops cannot press, so the voice looked healthy and never fired.)
    aliases = {
        "right_alt": "alt_r", "left_alt": "alt_l",
        "right_option": "alt_r", "left_option": "alt_l",
        "right_ctrl": "ctrl_r", "left_ctrl": "ctrl_l",
        "right_cmd": "cmd_r", "left_cmd": "cmd_l",
        "right_shift": "shift_r", "left_shift": "shift_l",
    }
    name = aliases.get(name, name)
    try:
        return getattr(keyboard.Key, name)
    except AttributeError:
        print(f"[ptt] unknown key {name!r} — falling back to 'home'",
              flush=True)
        return keyboard.Key.home


class PTTListener:
    def __init__(self, key="home"):
        self._key = resolve_key(key) if isinstance(key, str) else key
        self._held = False
        self._press_evt = threading.Event()
        self._listener = keyboard.Listener(on_press=self._on_press,
                                           on_release=self._on_release)
        self._listener.daemon = True
        self._listener.start()
        self._quartz_confirmed = False
        self._vk = self._physical_vk()
        if _HAS_QUARTZ and self._vk is not None:
            watchdog = threading.Thread(target=self._watchdog, daemon=True)
            watchdog.start()

    def _physical_vk(self):
        """macOS virtual keycode for our key, or None if unresolvable
        (e.g. a bare character key) — the watchdog is skipped then."""
        code = self._key.value if isinstance(self._key, keyboard.Key) \
            else self._key
        return getattr(code, "vk", None)

    def _watchdog(self):
        """Poll the real hardware state so a dropped/delayed OS event
        can never leave `_held` stuck out of sync (see module docstring).

        Only trusts a "not down" reading enough to clear a stuck `_held`
        once it has seen "down" agree with reality at least once — see
        the SELF-CONFIRMING GUARD note above. Without that gate, a read
        that never reports true (as seen on one Mac) would force-clear
        every real hold within one tick instead of only a stuck one."""
        while True:
            time.sleep(0.15)
            down = bool(CGEventSourceKeyState(
                kCGEventSourceStateHIDSystemState, self._vk))
            if down:
                self._quartz_confirmed = True
                if not self._held:
                    self._held = True
                    self._press_evt.set()
            elif self._held and self._quartz_confirmed:
                self._held = False

    def _on_press(self, k):
        if k == self._key and not self._held:   # filter key-repeat
            self._held = True
            self._press_evt.set()

    def _on_release(self, k):
        if k == self._key:
            self._held = False

    def wait_press(self):
        """Block until the key goes DOWN (one event per physical press)."""
        self._press_evt.wait()
        self._press_evt.clear()

    def is_held(self) -> bool:
        return self._held
