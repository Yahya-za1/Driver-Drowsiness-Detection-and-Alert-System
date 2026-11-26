"""
alert.py — Phase 4 alert module for the Driver Drowsiness Detection system.

Drives the red LED (GPIO27) and active buzzer (GPIO17, via BC547) with
four alert levels. Patterns run in a background thread, so calling
set_level() / alert() returns immediately and never blocks the camera loop.

Levels
  0  Normal       — everything off
  1  Warning      — LED slow blink (1 Hz), buzzer silent
  2  Drowsy       — LED + short beep together, 2 Hz
  3  Critical     — LED + beep, rapid 5 Hz alarm

Usage
  from alert import Alert
  a = Alert()
  a.set_level(2)
  ...
  a.close()

  # or the one-liner used by the rest of the system:
  from alert import alert
  alert(3)
"""

import atexit
import threading
import time

from gpiozero import OutputDevice

LED_PIN = 27      # physical pin 13
BUZZER_PIN = 17   # physical pin 11

# Each pattern is a list of (led_on, buzzer_on, duration_seconds) steps, looped.
PATTERNS = {
    0: [],
    1: [(True, False, 0.50), (False, False, 0.50)],
    2: [(True, True, 0.15), (False, False, 0.35)],
    3: [(True, True, 0.10), (False, False, 0.10)],
}


class Alert:
    def __init__(self, led_pin=LED_PIN, buzzer_pin=BUZZER_PIN):
        self.led = OutputDevice(led_pin, initial_value=False)
        self.buzzer = OutputDevice(buzzer_pin, initial_value=False)

        self._level = 0
        self._t_request = None          # when the latest level change was requested
        self._lock = threading.Lock()
        self._changed = threading.Event()
        self._stop = threading.Event()

        # latency log: (from_level, to_level, latency_ms)
        self.latencies = []

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # ---------- public API ----------
    @property
    def level(self):
        return self._level

    def set_level(self, level):
        level = int(level)
        if level not in PATTERNS:
            raise ValueError(f"level must be 0-3, got {level}")
        with self._lock:
            if level == self._level:
                return
            self._prev_level = self._level
            self._level = level
            self._t_request = time.perf_counter()
        self._changed.set()

    def close(self):
        self._stop.set()
        self._changed.set()
        self._thread.join(timeout=1)
        self._write(False, False)
        self.led.close()
        self.buzzer.close()

    # ---------- internals ----------
    def _write(self, led_on, buzzer_on):
        self.led.value = led_on
        self.buzzer.value = buzzer_on

    def _record_latency(self):
        with self._lock:
            if self._t_request is None:
                return
            ms = (time.perf_counter() - self._t_request) * 1000
            self.latencies.append((self._prev_level, self._level, ms))
            self._t_request = None

    def _run(self):
        while not self._stop.is_set():
            self._changed.clear()
            with self._lock:
                steps = PATTERNS[self._level]

            if not steps:                       # level 0: all off, wait for a change
                self._write(False, False)
                self._record_latency()
                self._changed.wait()
                continue

            first = True
            while not self._changed.is_set():
                for led_on, buzzer_on, duration in steps:
                    self._write(led_on, buzzer_on)
                    if first:
                        self._record_latency()
                        first = False
                    if self._changed.wait(duration):   # interrupted by a new level
                        break


# ---------- module-level convenience: alert(level) ----------
_default = None


def alert(level):
    global _default
    if _default is None:
        _default = Alert()
        atexit.register(_default.close)
    _default.set_level(level)


if __name__ == "__main__":
    # Quick self-test: step through every level for 3 s each.
    a = Alert()
    try:
        for lvl in (0, 1, 2, 3, 0):
            print(f"Level {lvl}")
            a.set_level(lvl)
            time.sleep(3)
        for frm, to, ms in a.latencies:
            print(f"  {frm} -> {to}: {ms:.3f} ms")
    finally:
        a.close()
