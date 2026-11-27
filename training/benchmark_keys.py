import sys
from pathlib import Path

import numpy as np
import torch
from doom_env import make_env

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))

from agent import DoomAgent

device = "cuda" if torch.cuda.is_available() else "cpu"

results = []

for path in sorted((ROOT / "maps" / "procedural").glob("*.wad")):
    env = make_env(
        str(path),
        "MAP01",
        episode_ticks=1400,
        render_buffers=False,
    )

    try:
        agent = DoomAgent(
            env.observation_space,
            env.action_space,
            device,
        )

        for seed in [42, 123, 456]:
            torch.manual_seed(seed)
            obs, _ = env.reset(seed=seed)
            agent.reset()

            key = False
            door = False
            exited = False

            for step in range(350):
                action = agent.act(obs)
                obs, reward, terminated, truncated, info = env.step(action)

                state = env.privileged_state()

                key |= bool(state["keys"].any())
                door |= bool(state["x"] > 448)
                exited |= bool(state["exit_reached"])

                if terminated or truncated:
                    break

            results.append((path.stem, key, door, exited))

            print(
                f"{path.stem:12} | Seed {seed:3} | "
                f"Key {key} | Door {door} | Exit {exited}",
                flush=True,
            )
    finally:
        env.close()

print("\nKEYCARD BENCHMARK COMPLETE")

for label, subset in [
    ("All maps", results),
    ("Training variants", [r for r in results if r[0].endswith(("_0", "_1"))]),
    ("Held-out variants", [r for r in results if r[0].endswith(("_2", "_3"))]),
]:
    if subset:
        data = np.array([r[1:] for r in subset], dtype=float)
        rates = data.mean(axis=0) * 100

        print(
            f"{label}: Episodes {len(subset)} | "
            f"Keys {rates[0]:.1f}% | "
            f"Doors {rates[1]:.1f}% | "
            f"Exits {rates[2]:.1f}%"
        )
