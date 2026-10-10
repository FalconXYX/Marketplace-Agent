"""Human pacing: a random pause before every browser action and a per-word typing speed."""

import random
import time
from collections.abc import Callable

KEY_DELAY_MS = (60, 180)


class Pacer:
    def __init__(
        self,
        delay_s: tuple[float, float],
        *,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
        key_delay_ms: tuple[int, int] = KEY_DELAY_MS,
    ):
        low, high = delay_s
        if low < 0 or high < low:
            raise ValueError(
                f"facebook_action_delay_s must be (low, high) with 0 <= low <= high, got {delay_s}"
            )
        self.delay_s = (low, high)
        self.key_delay_range = key_delay_ms
        self._sleep = sleep
        self._rng = rng or random.Random()

    def pause(self) -> float:
        """Sleep a random time in delay_s; returns it."""
        seconds = self._rng.uniform(*self.delay_s)
        self._sleep(seconds)
        return seconds

    def key_delay_ms(self) -> int:
        """Delay between keystrokes for the next word."""
        return self._rng.randint(*self.key_delay_range)
