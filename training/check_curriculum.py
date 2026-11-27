
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

from training.curriculum import (
    CURRICULUM,
    BENCHMARK_MAPS,
    CurriculumEnv,
    build_env
)


rng = np.random.default_rng(42)

passed = 0

failed = []

total_steps = 0

start = time.perf_counter()


def check_map(spec, shaped=True):

    global passed, total_steps

    env = None

    try:

        env = build_env(
            spec,
            shaped=shaped
        )

        obs, info = env.reset(seed=42)

        assert obs["rays"]["distance"].shape == (64,)

        assert obs["vitals"].shape == (17,)

        assert obs["visible_entities"]["features"].shape == (
            32, 5
        )

        assert env.action_space.shape == (9,)

        entity_count = int(
            obs["visible_entities"]["mask"].sum()
        )

        total_reward = 0.0

        for step in range(32):

            action = rng.integers(
                0, 2,
                size=9,
                dtype=np.int8
            )

            obs, reward, terminated, truncated, info = (
                env.step(action)
            )

            assert np.isfinite(reward)

            total_reward += reward

            total_steps += 1

            if terminated or truncated:

                obs, info = env.reset()

        print(
            f"PASS | {spec.name:25s} "
            f"| Entities: {entity_count:2d} "
            f"| Reward: {total_reward:+.3f}"
        )

        passed += 1

    except Exception as error:

        failed.append(
            (spec.name, str(error))
        )

        print(
            f"FAIL | {spec.name}: {error}"
        )

    finally:

        if env is not None:

            env.close()


print("\nPROJECT-DOOM CURRICULUM VALIDATION")
print("=" * 60)


for stage, maps in CURRICULUM.items():

    print(f"\nCURRICULUM STAGE {stage}")
    print("=" * 60)

    for spec in maps:

        check_map(spec)


print("\nHELD-OUT BENCHMARK MAPS")
print("=" * 60)

for spec in BENCHMARK_MAPS:

    check_map(spec, shaped=False)


print("\nCURRICULUM SWITCHING TEST")
print("=" * 60)

env = None

try:

    env = CurriculumEnv(stage=1, seed=42)

    for stage in range(1, 6):

        env.set_stage(stage)

        obs, info = env.reset()

        assert info["curriculum_stage"] == stage

        print(
            f"Stage {stage}: "
            f"{info['map_name']} - PASS"
        )

except Exception as error:

    failed.append(
        ("Curriculum switching", str(error))
    )

finally:

    if env is not None:

        env.close()


elapsed = time.perf_counter() - start

print("\nFINAL RESULTS")
print("=" * 60)

print("Maps passed:", passed)

print("Failures:", len(failed))

print("Total decisions:", total_steps)

print(f"Elapsed time: {elapsed:.2f}s")


if failed:

    print("\nFAILED MAPS")

    for name, error in failed:

        print(f"{name}: {error}")

else:

    print("\nALL CURRICULUM TESTS PASSED")
