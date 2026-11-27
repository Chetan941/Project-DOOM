
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import gymnasium as gym
import vizdoom

from doom_env import make_env
from training.reward_wrapper import DoomRewardWrapper


ROOT = Path(__file__).resolve().parents[1]

VIZDOOM_DIR = Path(vizdoom.__file__).resolve().parent

SCENARIOS = VIZDOOM_DIR / "scenarios"

FREEDOOM = VIZDOOM_DIR / "freedoom2.wad"


@dataclass(frozen=True)
class MapSpec:

    name: str
    path: Path
    map_id: str = "MAP01"
    episode_ticks: int = 2100


def scenario(name, ticks=2100):

    return MapSpec(
        name=name,
        path=SCENARIOS / f"{name}.wad",
        episode_ticks=ticks
    )


def freedoom(number, ticks=4200):

    map_id = f"MAP{number:02d}"

    return MapSpec(
        name=f"freedoom_{map_id}",
        path=FREEDOOM,
        map_id=map_id,
        episode_ticks=ticks
    )


TWO_ROOMS = MapSpec(
    name="two_rooms",
    path=ROOT / "maps" / "two_rooms.wad"
)


CURRICULUM = {

    1: [
        scenario("basic", 1050),
        scenario("simpler_basic", 1050)
    ],

    2: [
        scenario("my_way_home"),
        TWO_ROOMS
    ],

    3: [
        scenario("deadly_corridor"),
        scenario("defend_the_line"),
        scenario("defend_the_center")
    ],

    4: [
        freedoom(i)
        for i in range(1, 7)
    ],

    5: [
        freedoom(i)
        for i in range(7, 17)
    ]
}


BENCHMARK_MAPS = [
    freedoom(i)
    for i in range(17, 21)
]


def build_env(spec, shaped=True):

    if not spec.path.exists():

        raise FileNotFoundError(spec.path)

    env = make_env(
        str(spec.path),
        spec.map_id,
        episode_ticks=spec.episode_ticks,
        render_buffers=False
    )

    if shaped:

        env = DoomRewardWrapper(env)

    return env


class CurriculumEnv(gym.Env):

    def __init__(self, stage=1, seed=0):

        super().__init__()

        self.stage = stage

        self.rng = np.random.default_rng(seed)

        self.active_env = None

        self.active_spec = None

        self.active_stage = None

        self._select_environment()

        self.observation_space = (
            self.active_env.observation_space
        )

        self.action_space = (
            self.active_env.action_space
        )

    def set_stage(self, stage):

        if stage not in CURRICULUM:

            raise ValueError(
                f"Invalid curriculum stage: {stage}"
            )

        self.stage = stage

    def _select_environment(self):

        pool = CURRICULUM[self.stage]

        index = int(
            self.rng.integers(len(pool))
        )

        spec = pool[index]

        if (
            self.active_env is not None
            and self.active_spec == spec
            and self.active_stage == self.stage
        ):

            return

        if self.active_env is not None:

            self.active_env.close()

        self.active_env = build_env(spec)

        self.active_spec = spec

        self.active_stage = self.stage

    def reset(self, *, seed=None, options=None):

        super().reset(seed=seed)

        if seed is not None:

            self.rng = np.random.default_rng(seed)

        self._select_environment()

        episode_seed = int(
            self.rng.integers(0, 2**31 - 1)
        )

        observation, info = self.active_env.reset(
            seed=episode_seed
        )

        info = dict(info)

        info["map_name"] = self.active_spec.name

        info["curriculum_stage"] = self.active_stage

        return observation, info

    def step(self, action):

        obs, reward, terminated, truncated, info = (
            self.active_env.step(action)
        )

        info = dict(info)

        info["map_name"] = self.active_spec.name

        info["curriculum_stage"] = self.active_stage

        return (
            obs,
            reward,
            terminated,
            truncated,
            info
        )

    def close(self):

        if self.active_env is not None:

            self.active_env.close()

            self.active_env = None
