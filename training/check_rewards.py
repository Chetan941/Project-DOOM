
import sys
import time
from pathlib import Path
from collections import Counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

from doom_env import make_env
from training.reward_wrapper import DoomRewardWrapper


base_env = make_env(
    str(ROOT / "maps" / "two_rooms.wad"),
    "MAP01",
    episode_ticks=2100,
    render_buffers=False
)

env = DoomRewardWrapper(base_env)

rng = np.random.default_rng(42)

totals = Counter()

try:

    obs, info = env.reset(seed=42)

    print("\nREWARD SYSTEM VALIDATION")
    print("=" * 60)

    state = env.get_state()

    print("\nPRIVILEGED TRAINING VARIABLES")
    print("=" * 60)

    for key, value in state.items():

        if isinstance(value, np.ndarray):

            print(
                f"{key}: shape={value.shape}"
            )

        else:

            print(f"{key}: {value}")

    print("\nRUNNING 1000 GAME DECISIONS")
    print("=" * 60)

    start = time.perf_counter()

    original_total = 0.0
    shaped_total = 0.0

    episodes = 0
    exits = 0
    deaths = 0
    timeouts = 0

    for step in range(1000):

        action = rng.integers(
            0, 2,
            size=9,
            dtype=np.int8
        )

        obs, reward, terminated, truncated, info = (
            env.step(action)
        )

        assert np.isfinite(reward)

        components = info["reward_components"]

        assert np.isclose(
            reward,
            sum(components.values())
        )

        totals.update(components)

        original_total += info[
            "original_reward"
        ]

        shaped_total += reward

        if (step + 1) % 100 == 0:

            print(
                f"Step {step + 1}: "
                f"Reward={reward:.4f}, "
                f"Cells={info['visited_cells']}, "
                f"Doors={info['opened_doors']}, "
                f"Kills={info['kills']}"
            )

        if terminated or truncated:

            episodes += 1

            exits += int(info["exit_reached"])

            deaths += int(info["dead"])

            timeouts += int(truncated)

            obs, info = env.reset()

    elapsed = time.perf_counter() - start

    print("\nREWARD COMPONENT TOTALS")
    print("=" * 60)

    for key, value in totals.items():

        print(f"{key:20s}: {value:+.6f}")

    print("\nEPISODE STATISTICS")
    print("=" * 60)

    print("Completed episodes:", episodes)
    print("Exits reached:", exits)
    print("Deaths:", deaths)
    print("Timeouts:", timeouts)

    print(
        f"Original total reward: "
        f"{original_total:.6f}"
    )

    print(
        f"Shaped total reward: "
        f"{shaped_total:.6f}"
    )

    print(
        f"Environment FPS: "
        f"{1000 / elapsed:.2f}"
    )

    print("\nMEMORY RESET TEST")
    print("=" * 60)

    obs, info = env.reset(seed=42)

    assert len(env.visited_cells) == 1

    assert len(env.observation_counts) == 1

    assert len(env.opened_doors) == 0

    assert len(env.reward_totals) == 0

    print("Exploration memory reset: Passed")
    print("Curiosity memory reset: Passed")
    print("Door tracking reset: Passed")
    print("Reward statistics reset: Passed")

    print("\nALL REWARD TESTS PASSED")

finally:

    env.close()
