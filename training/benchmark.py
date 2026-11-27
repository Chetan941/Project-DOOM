import sys
import time
import csv
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from submission.agent import DoomAgent
from training.curriculum import (
    TWO_ROOMS,
    BENCHMARK_MAPS,
    scenario,
    build_env,
)

torch.set_num_threads(1)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [42, 123, 456]

MAPS = [
    scenario("basic", 1050),
    scenario("my_way_home"),
    TWO_ROOMS,
    *BENCHMARK_MAPS,
]

RESULTS = []


def evaluate(spec, policy, seed):
    env = build_env(spec, shaped=False)

    try:
        obs, info = env.reset(seed=seed)

        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        agent = None

        if policy == "trained":
            agent = DoomAgent(
                env.observation_space,
                env.action_space,
                DEVICE,
            )
            agent.reset()

        cells = set()
        latencies = []
        total_reward = 0.0
        steps = 0

        def record_cell(state):
            cells.add((
                int(np.floor(state["x"] / 64)),
                int(np.floor(state["y"] / 64)),
            ))

        record_cell(env.unwrapped.privileged_state())

        while True:
            start = time.perf_counter()

            if policy == "trained":
                action = agent.act(obs)
            else:
                action = rng.integers(
                    0, 2, size=9, dtype=np.int8
                )

            latencies.append(
                (time.perf_counter() - start) * 1000
            )

            obs, reward, terminated, truncated, info = (
                env.step(action)
            )

            steps += 1
            total_reward += reward

            state = env.unwrapped.privileged_state()
            record_cell(state)

            if terminated or truncated:
                break

        result = {
            "map": spec.name,
            "policy": policy,
            "seed": seed,
            "steps": steps,
            "reward": round(total_reward, 4),
            "exit": int(state["exit_reached"]),
            "dead": int(state["dead"]),
            "timeout": int(state["timeout"]),
            "kills": state["kills"],
            "keys": int(np.sum(state["keys"])),
            "cells": len(cells),
            "health": round(state["health"], 2),
            "damage": round(state["damage_taken"], 2),
            "mean_latency_ms": round(
                float(np.mean(latencies)), 3
            ),
            "p99_latency_ms": round(
                float(np.percentile(latencies, 99)), 3
            ),
        }

        return result

    finally:
        env.close()


def main():
    start = time.perf_counter()

    print("\nDOOM CORTEX BENCHMARK")
    print("Device:", DEVICE)
    print("Maps:", len(MAPS))
    print("Seeds:", len(SEEDS))
    print("Total episodes:", len(MAPS) * len(SEEDS) * 2)

    for spec in MAPS:
        print(f"\nEvaluating: {spec.name}")

        for policy in ["random", "trained"]:
            for seed in SEEDS:
                result = evaluate(spec, policy, seed)
                RESULTS.append(result)

                print(
                    f"{policy:8s} | "
                    f"Seed {seed:3d} | "
                    f"Exit {result['exit']} | "
                    f"Kills {result['kills']:3d} | "
                    f"Keys {result['keys']} | "
                    f"Cells {result['cells']:4d} | "
                    f"Health {result['health']:6.1f} | "
                    f"Reward {result['reward']:+.2f}",
                    flush=True,
                )

    output = ROOT / "training" / "logs" / "benchmark.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(RESULTS[0].keys())
        )
        writer.writeheader()
        writer.writerows(RESULTS)

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    for spec in MAPS:
        print(f"\nMAP: {spec.name}")

        for policy in ["random", "trained"]:
            rows = [
                r for r in RESULTS
                if r["map"] == spec.name
                and r["policy"] == policy
            ]

            def avg(key):
                return np.mean([r[key] for r in rows])

            print(
                f"{policy:8s} | "
                f"Exits {avg('exit') * 100:5.1f}% | "
                f"Kills {avg('kills'):6.2f} | "
                f"Keys {avg('keys'):4.2f} | "
                f"Cells {avg('cells'):7.2f} | "
                f"Health {avg('health'):6.1f} | "
                f"Reward {avg('reward'):+.3f}"
            )

    print("\nBENCHMARK COMPLETE")
    print("Results:", output)
    print(
        f"Elapsed: {(time.perf_counter() - start) / 60:.2f} minutes"
    )


if __name__ == "__main__":
    main()
import time
import csv
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from submission.agent import DoomAgent
from training.curriculum import (
    TWO_ROOMS,
    BENCHMARK_MAPS,
    scenario,
    build_env,
)

torch.set_num_threads(1)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [42, 123, 456]

MAPS = [
    scenario("basic", 1050),
    scenario("my_way_home"),
    TWO_ROOMS,
    *BENCHMARK_MAPS,
]

RESULTS = []


def evaluate(spec, policy, seed):
    env = build_env(spec, shaped=False)

    try:
        obs, info = env.reset(seed=seed)

        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        agent = None

        if policy == "trained":
            agent = DoomAgent(
                env.observation_space,
                env.action_space,
                DEVICE,
            )
            agent.reset()

        cells = set()
        latencies = []
        total_reward = 0.0
        steps = 0

        def record_cell(state):
            cells.add((
                int(np.floor(state["x"] / 64)),
                int(np.floor(state["y"] / 64)),
            ))

        record_cell(env.unwrapped.privileged_state())

        while True:
            start = time.perf_counter()

            if policy == "trained":
                action = agent.act(obs)
            else:
                action = rng.integers(
                    0, 2, size=9, dtype=np.int8
                )

            latencies.append(
                (time.perf_counter() - start) * 1000
            )

            obs, reward, terminated, truncated, info = (
                env.step(action)
            )

            steps += 1
            total_reward += reward

            state = env.unwrapped.privileged_state()
            record_cell(state)

            if terminated or truncated:
                break

        result = {
            "map": spec.name,
            "policy": policy,
            "seed": seed,
            "steps": steps,
            "reward": round(total_reward, 4),
            "exit": int(state["exit_reached"]),
            "dead": int(state["dead"]),
            "timeout": int(state["timeout"]),
            "kills": state["kills"],
            "keys": int(np.sum(state["keys"])),
            "cells": len(cells),
            "health": round(state["health"], 2),
            "damage": round(state["damage_taken"], 2),
            "mean_latency_ms": round(
                float(np.mean(latencies)), 3
            ),
            "p99_latency_ms": round(
                float(np.percentile(latencies, 99)), 3
            ),
        }

        return result

    finally:
        env.close()


def main():
    start = time.perf_counter()

    print("\nDOOM CORTEX BENCHMARK")
    print("Device:", DEVICE)
    print("Maps:", len(MAPS))
    print("Seeds:", len(SEEDS))
    print("Total episodes:", len(MAPS) * len(SEEDS) * 2)

    for spec in MAPS:
        print(f"\nEvaluating: {spec.name}")

        for policy in ["random", "trained"]:
            for seed in SEEDS:
                result = evaluate(spec, policy, seed)
                RESULTS.append(result)

                print(
                    f"{policy:8s} | "
                    f"Seed {seed:3d} | "
                    f"Exit {result['exit']} | "
                    f"Kills {result['kills']:3d} | "
                    f"Keys {result['keys']} | "
                    f"Cells {result['cells']:4d} | "
                    f"Health {result['health']:6.1f} | "
                    f"Reward {result['reward']:+.2f}",
                    flush=True,
                )

    output = ROOT / "training" / "logs" / "benchmark.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(RESULTS[0].keys())
        )
        writer.writeheader()
        writer.writerows(RESULTS)

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    for spec in MAPS:
        print(f"\nMAP: {spec.name}")

        for policy in ["random", "trained"]:
            rows = [
                r for r in RESULTS
                if r["map"] == spec.name
                and r["policy"] == policy
            ]

            def avg(key):
                return np.mean([r[key] for r in rows])

            print(
                f"{policy:8s} | "
                f"Exits {avg('exit') * 100:5.1f}% | "
                f"Kills {avg('kills'):6.2f} | "
                f"Keys {avg('keys'):4.2f} | "
                f"Cells {avg('cells'):7.2f} | "
                f"Health {avg('health'):6.1f} | "
                f"Reward {avg('reward'):+.3f}"
            )

    print("\nBENCHMARK COMPLETE")
    print("Results:", output)
    print(
        f"Elapsed: {(time.perf_counter() - start) / 60:.2f} minutes"
    )


if __name__ == "__main__":
    main()