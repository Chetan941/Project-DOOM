
from __future__ import annotations

from collections import Counter

import gymnasium as gym
import numpy as np


class DoomRewardWrapper(gym.Wrapper):

    def __init__(
        self,
        env,
        curiosity_beta=0.003,
        cell_size=64.0
    ):

        super().__init__(env)

        self.curiosity_beta = curiosity_beta
        self.cell_size = cell_size

        self.visited_cells = set()
        self.observation_counts = Counter()
        self.opened_doors = set()

        self.previous = None

        self.reward_totals = Counter()

    def get_state(self):

        return self.env.unwrapped.privileged_state()

    def spatial_cell(self, state):

        return (
            int(np.floor(
                state["x"] / self.cell_size
            )),
            int(np.floor(
                state["y"] / self.cell_size
            ))
        )

    def observation_hash(self, observation):

        distances = np.asarray(
            observation["rays"]["distance"]
        )

        hit_types = np.asarray(
            observation["rays"]["hit_type"]
        )

        quantized = np.floor(
            distances * 16
        ).astype(np.int32)

        return (
            quantized.tobytes(),
            hit_types.astype(np.int8).tobytes()
        )

    def reset(self, **kwargs):

        observation, info = self.env.reset(
            **kwargs
        )

        self.previous = self.get_state()

        self.visited_cells.clear()
        self.observation_counts.clear()
        self.opened_doors.clear()
        self.reward_totals.clear()

        self.visited_cells.add(
            self.spatial_cell(self.previous)
        )

        signature = self.observation_hash(
            observation
        )

        self.observation_counts[signature] = 1

        return observation, info

    def step(self, action):

        observation, original_reward, terminated, truncated, info = (
            self.env.step(action)
        )

        current = self.get_state()
        previous = self.previous

        components = {}

        # Time penalty

        components["time"] = -0.005

        # Combat

        kills = max(
            0,
            current["kills"] - previous["kills"]
        )

        components["combat"] = float(kills)

        # Damage and survival

        damage = max(
            0.0,
            current["damage_taken"]
            - previous["damage_taken"]
        )

        components["damage"] = -0.02 * damage

        recovered = max(
            0.0,
            current["health"] - previous["health"]
        )

        components["health"] = 0.01 * recovered

        # Item collection

        items = max(
            0,
            current["items"] - previous["items"]
        )

        components["items"] = 0.10 * items

        # Keycard collection

        new_keys = np.logical_and(
            np.asarray(current["keys"]) > 0,
            np.asarray(previous["keys"]) == 0
        )

        components["keys"] = (
            3.0 * float(new_keys.sum())
        )

        # Door-opening progress

        door_reward = 0.0

        old_ceiling = previous["sector_ceiling"]
        new_ceiling = current["sector_ceiling"]

        old_floor = previous["sector_floor"]
        locks = previous["door_lock"]

        if (
            old_ceiling is not None
            and new_ceiling is not None
            and old_floor is not None
            and locks is not None
        ):

            old_ceiling = np.asarray(old_ceiling)
            new_ceiling = np.asarray(new_ceiling)
            old_floor = np.asarray(old_floor)
            locks = np.asarray(locks)

            if (
                old_ceiling.shape
                == new_ceiling.shape
                == old_floor.shape
                == locks.shape
            ):

                opening = np.flatnonzero(
                    (old_ceiling - old_floor <= 8)
                    & (new_ceiling > old_ceiling)
                )

                for index in opening:

                    index = int(index)

                    if index not in self.opened_doors:

                        self.opened_doors.add(index)

                        door_reward += 1.0

        components["doors"] = door_reward

        # Spatial exploration

        cell = self.spatial_cell(current)

        exploration_reward = 0.0

        if cell not in self.visited_cells:

            self.visited_cells.add(cell)

            exploration_reward = 0.05

        components["exploration"] = (
            exploration_reward
        )

        # Episodic observation novelty

        curiosity = 0.0

        if not (terminated or truncated):

            signature = self.observation_hash(
                observation
            )

            self.observation_counts[signature] += 1

            visits = self.observation_counts[
                signature
            ]

            curiosity = (
                self.curiosity_beta
                / np.sqrt(visits)
            )

        components["curiosity"] = float(curiosity)

        # Terminal objectives

        components["exit"] = (
            10.0 if current["exit_reached"] else 0.0
        )

        components["death"] = (
            -3.0 if current["dead"] else 0.0
        )

        # Combine training rewards

        reward = float(
            sum(components.values())
        )

        self.reward_totals.update(components)

        self.previous = current

        info = dict(info)

        info["original_reward"] = float(
            original_reward
        )

        info["reward_components"] = components

        info["reward_totals"] = dict(
            self.reward_totals
        )

        info["visited_cells"] = len(
            self.visited_cells
        )

        info["unique_observations"] = len(
            self.observation_counts
        )

        info["opened_doors"] = len(
            self.opened_doors
        )

        info["kills"] = current["kills"]

        info["health"] = current["health"]

        info["keys_collected"] = int(
            np.asarray(current["keys"]).sum()
        )

        info["exit_reached"] = current[
            "exit_reached"
        ]

        info["dead"] = current["dead"]

        return (
            observation,
            reward,
            terminated,
            truncated,
            info
        )
