import asyncio
import time

from vlnav.config import Config
from vlnav.sim_interface import SimInterface


class FakeDrone:
    """Mimics projectairsim: the first await returns quickly with a task that completes later."""
    async def move_by_velocity_async(self, *a, **k):
        return asyncio.create_task(asyncio.sleep(0.3))


def test_velocity_nowait_returns_fast_and_tracks_pending():
    async def go():
        sim = SimInterface(Config()); sim.drone = FakeDrone()
        t0 = time.monotonic()
        task = await sim.velocity_nowait(1.0, 0.0, 0.0, 0.15)
        fast = time.monotonic() - t0
        assert fast < 0.1 and sim.pending_commands == 1 and not task.done()
        await asyncio.sleep(0.4)
        assert sim.pending_commands == 0 and task.done()
        t1 = time.monotonic()
        await sim.velocity(1.0, 0.0, 0.0, 0.15)          # the old path blocks until completion
        assert time.monotonic() - t1 >= 0.29
    asyncio.run(go())


def test_many_nowait_commands_do_not_serialise():
    async def go():
        sim = SimInterface(Config()); sim.drone = FakeDrone()
        t0 = time.monotonic()
        for _ in range(20):
            await sim.velocity_nowait(1.0, 0.0, 0.0, 0.15)
        assert time.monotonic() - t0 < 0.3 and sim.pending_commands > 0   # 20 sends << 20 x 0.3 s
        await asyncio.sleep(0.4)
        assert sim.pending_commands == 0
    asyncio.run(go())
