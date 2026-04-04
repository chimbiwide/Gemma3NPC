import asyncio
import time

class RateLimiter:
    def __init__(self, max_per_min: int):
        self.max_rpm = max_per_min
        self._lock = asyncio.Lock()
        self._timestamps: list[float] = []

    async def wait(self):
        async with self._lock:
            now = time.monotonic()
            self._timestamps = [t for t in self._timestamps if now - t < 60]
            if len(self._timestamps) >= self.max_rpm:
                sleep_time = 60 - (now - self._timestamps[0])
                await asyncio.sleep(sleep_time)
            self._timestamps.append(time.monotonic())
