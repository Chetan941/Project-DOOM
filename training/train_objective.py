
from __future__ import annotations

import multiprocessing as mp
import traceback
import argparse
import csv
import sys
import time
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from submission.model import DoomCortex
from training.curriculum import CurriculumEnv


def worker(conn, index, seed):
    env = None
    try:
        from training.curriculum import TWO_ROOMS, scenario, freedoom, build_env
        # Fixed map per process avoids repeated WAD initialization.
        from doom_env.wad import read_wad, map_lumps
        import struct
        candidates = [freedoom(i) for i in range(1, 17)]
        lumps = read_wad(candidates[0].path)
        keyed = []
        for candidate in candidates:
            data = map_lumps(lumps, candidate.map_id)
            if "TEXTMAP" in data or "BEHAVIOR" in data:
                continue
            things = data.get("THINGS", b"")
            if len(things) % 10 == 0 and any(t[3] in (5, 6, 13, 38, 39, 40) for t in struct.iter_unpack("<hhhhh", things)):
                keyed.append(candidate)
        if not keyed:
            raise RuntimeError("No Doom-format keycard training maps found")
        maps = [TWO_ROOMS, scenario("my_way_home")] + [keyed[i % len(keyed)] for i in range(6)]
        spec = maps[index % len(maps)]
        env = build_env(spec)
        print(f"Worker {index}: {spec.name}", flush=True)
        def key_distance(obs):
            e = obs["visible_entities"]
            mask = (e["entity_type"] == 9) & (e["mask"] != 0)
            return float(np.min(e["features"][mask, 0])) if mask.any() else None
        previous_key = None
        rng = np.random.default_rng(seed)
        def reset():
            nonlocal previous_key
            obs, info = env.reset(seed=int(rng.integers(0, 2**31-1)))
            previous_key = key_distance(obs)
            return slim(obs), info
        conn.send((True, reset()))
        while True:
            cmd, action = conn.recv()
            if cmd == "close":
                break
            obs, reward, term, trunc, info = env.step(action)
            c = info["reward_components"]
            # Training-only shaping: stronger exploration and objective progress.
            reward += c["exploration"] + 3*c["doors"] + 3*c["keys"]
            reward += 2*c["damage"] + c["death"] + c["exit"]
            current_key = key_distance(obs)
            if not (term or trunc) and previous_key is not None and current_key is not None and not c["keys"]:
                reward += float(np.clip(4*(previous_key-current_key), -0.2, 0.2))
            previous_key = current_key
            result = (slim(obs), reward, term, trunc, info)
            fresh = reset() if term or trunc else None
            conn.send((True, (result, fresh)))
    except (EOFError, KeyboardInterrupt):
        pass
    except Exception:
        try:
            conn.send((False, traceback.format_exc()))
        except Exception:
            pass
    finally:
        if env is not None:
            env.close()
        conn.close()


def slim(obs):
    return {k: obs[k] for k in ("rays", "vitals", "visible_entities")}


class ParallelEnv:
    def __init__(self, index, seed):
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.process = ctx.Process(target=worker, args=(child, index, seed), daemon=True)
        self.process.start()
        child.close()
        self.cached = None

    def receive(self):
        if not self.conn.poll(120):
            raise RuntimeError("Environment worker timed out")
        ok, data = self.conn.recv()
        if not ok:
            raise RuntimeError(data)
        return data

    def reset(self):
        if self.cached is None:
            return self.receive()
        result, self.cached = self.cached, None
        return result

    def set_stage(self, stage):
        pass

    def send_step(self, action):
        self.conn.send(("step", action))

    def step(self, action):
        result, self.cached = self.receive()
        return result

    def close(self):
        try:
            self.conn.send(("close", None))
        except (OSError, EOFError):
            pass
        self.process.join(2)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()
        self.conn.close()


def main():
    # ==========================================
    # CONFIGURATION
    # ==========================================

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--steps", type=int, default=100000
    )

    parser.add_argument(
        "--envs", type=int, default=8
    )

    parser.add_argument(
        "--rollout", type=int, default=128
    )

    parser.add_argument(
        "--device", default="cuda"
    )

    parser.add_argument(
        "--resume", action="store_true"
    )

    parser.add_argument(
        "--seed", type=int, default=42
    )

    args = parser.parse_args()

    N = args.envs
    T = args.rollout

    SEQ = 16
    BATCH_SEQUENCES = 16
    EPOCHS = 4

    LR = 5e-5
    GAMMA = 0.99
    GAE_LAMBDA = 0.95

    CLIP = 0.2
    ENTROPY_COEF = 0.02
    VALUE_COEF = 0.5

    assert T % SEQ == 0

    torch.set_num_threads(1)

    torch.manual_seed(args.seed)

    np.random.seed(args.seed)

    device = torch.device(
        args.device
        if torch.cuda.is_available()
        else "cpu"
    )

    WEIGHTS = ROOT / "submission" / "weights"

    WEIGHTS.mkdir(
        parents=True, exist_ok=True
    )

    CHECKPOINT = WEIGHTS / "doom_cortex_objective.pt"

    LOG_DIR = ROOT / "training" / "logs"

    LOG_DIR.mkdir(
        parents=True, exist_ok=True
    )

    LOG_FILE = LOG_DIR / "training_objective.csv"


    # ==========================================
    # OBSERVATION PROCESSING
    # ==========================================

    def compact_obs(obs):

        return {
            "distance": obs["rays"]["distance"],
            "hit_type": obs["rays"]["hit_type"],
            "vitals": obs["vitals"],
            "features": obs["visible_entities"]["features"],
            "entity_type": obs["visible_entities"]["entity_type"],
            "mask": obs["visible_entities"]["mask"]
        }


    def stack_obs(observations):

        compact = [
            compact_obs(o)
            for o in observations
        ]

        return {
            key: torch.as_tensor(
                np.stack([
                    o[key] for o in compact
                ]),
                device=device
            )
            for key in compact[0]
        }


    def model_obs(data):

        return {
            "rays": {
                "distance": data["distance"],
                "hit_type": data["hit_type"]
            },

            "vitals": data["vitals"],

            "visible_entities": {
                "features": data["features"],
                "entity_type": data["entity_type"],
                "mask": data["mask"]
            }
        }


    # ==========================================
    # MODEL AND OPTIMIZER
    # ==========================================

    model = DoomCortex().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        eps=1e-5
    )

    global_step = 0

    if args.resume:

        checkpoint = torch.load(
            CHECKPOINT if CHECKPOINT.exists() else WEIGHTS / "doom_cortex_parallel.pt",
            map_location=device,
            weights_only=True
        )

        model.load_state_dict(
            checkpoint["model"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        for group in optimizer.param_groups:
            group["lr"] = LR

        global_step = checkpoint["steps"]

        print(
            f"Resumed from {global_step} decisions"
        )

    target_step = global_step + args.steps


    # ==========================================
    # CURRICULUM SCHEDULE
    # ==========================================

    def curriculum_stage(step):

        if step < 20000:
            return 1

        if step < 40000:
            return 2

        if step < 60000:
            return 3

        if step < 100000:
            return 4

        return 5


    def stage_for_env(index, stage):

        # Retain experience from earlier stages
        # to reduce catastrophic forgetting.

        if index == 0 and stage > 1:
            return 1

        if index == 1 and stage > 2:
            return 2

        return stage


    # ==========================================
    # CHECKPOINT SAVING
    # ==========================================

    def save_checkpoint():

        temporary = CHECKPOINT.with_suffix(
            ".tmp"
        )

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "steps": global_step
            },
            temporary
        )

        temporary.replace(CHECKPOINT)


    # ==========================================
    # ENVIRONMENT INITIALIZATION
    # ==========================================

    envs = []

    observations = []

    episode_rewards = np.zeros(
        N, dtype=np.float32
    )

    recent_rewards = deque(maxlen=100)
    recent_kills = deque(maxlen=100)
    recent_exits = deque(maxlen=100)
    recent_keys = deque(maxlen=100)
    recent_cells = deque(maxlen=100)
    recent_doors = deque(maxlen=100)

    episodes = 0

    hidden = model.initial_state(N)

    previous_action = torch.zeros(
        N, 9, device=device
    )

    start_time = time.perf_counter()

    initial_step = global_step

    last_checkpoint = global_step

    update = 0


    try:

        for i in range(N):

            stage = stage_for_env(
                i,
                curriculum_stage(global_step)
            )

            env = ParallelEnv(i, args.seed + i)

            envs.append(env)

        for env in envs:
            obs, info = env.reset()
            observations.append(obs)

        print("\nPROJECT-DOOM TRAINING")
        print("=" * 65)

        print("Device:", device)

        print("Environments:", N)

        print("Target decisions:", target_step)

        print(
            "Trainable parameters:",
            sum(p.numel() for p in model.parameters())
        )

        # ======================================
        # MAIN TRAINING LOOP
        # ======================================

        while global_step < target_step:

            update += 1

            stage = curriculum_stage(
                global_step
            )

            for i, env in enumerate(envs):

                env.set_stage(
                    stage_for_env(i, stage)
                )

            obs_buffer = {}

            hidden_buffer = []

            previous_buffer = []

            action_buffer = []

            logprob_buffer = []

            value_buffer = []

            reward_buffer = []

            done_buffer = []

            timeout_value_buffer = []

            model.eval()

            # ==================================
            # ROLLOUT COLLECTION
            # ==================================

            with torch.no_grad():

                for t in range(T):

                    data = stack_obs(
                        observations
                    )

                    for key, value in data.items():

                        obs_buffer.setdefault(
                            key, []
                        ).append(value.clone())

                    hidden_buffer.append(
                        hidden.clone()
                    )

                    previous_buffer.append(
                        previous_action.clone()
                    )

                    logits, values, next_hidden = model(
                        model_obs(data),
                        hidden,
                        previous_action
                    )

                    distribution = (
                        torch.distributions.Bernoulli(
                            logits=logits
                        )
                    )

                    actions = distribution.sample()

                    logprobs = distribution.log_prob(
                        actions
                    ).sum(dim=-1)

                    action_buffer.append(
                        actions.clone()
                    )

                    logprob_buffer.append(
                        logprobs.clone()
                    )

                    value_buffer.append(
                        values.clone()
                    )

                    numpy_actions = (
                        actions.cpu()
                        .numpy()
                        .astype(np.int8)
                    )

                    rewards = np.zeros(
                        N, dtype=np.float32
                    )

                    dones = np.zeros(
                        N, dtype=np.float32
                    )

                    timeout_values = torch.zeros(
                        N, device=device
                    )

                    new_observations = []

                    for i, env in enumerate(envs):
                        env.send_step(numpy_actions[i])

                    for i, env in enumerate(envs):

                        obs, reward, terminated, truncated, info = (
                            env.step(numpy_actions[i])
                        )

                        rewards[i] = reward

                        episode_rewards[i] += reward

                        done = terminated or truncated

                        dones[i] = float(done)

                        # Bootstrap time-limit truncations
                        # before resetting the environment.

                        if truncated and not terminated:

                            terminal_data = stack_obs(
                                [obs]
                            )

                            _, terminal_value, _ = model(
                                model_obs(terminal_data),
                                next_hidden[i:i+1],
                                actions[i:i+1]
                            )

                            timeout_values[i] = (
                                terminal_value[0]
                            )

                        if done:

                            episodes += 1

                            recent_rewards.append(
                                float(episode_rewards[i])
                            )

                            recent_kills.append(
                                info.get("kills", 0)
                            )

                            recent_exits.append(
                                int(info.get("exit_reached", False))
                            )

                            recent_keys.append(
                                info.get("keys_collected", 0)
                            )

                            recent_cells.append(
                                info.get("visited_cells", 0)
                            )

                            recent_doors.append(
                                info.get("opened_doors", 0)
                            )

                            episode_rewards[i] = 0

                            obs, _ = env.reset()

                        new_observations.append(obs)

                    observations = new_observations

                    reward_buffer.append(
                        torch.as_tensor(
                            rewards,
                            device=device
                        )
                    )

                    done_tensor = torch.as_tensor(
                        dones,
                        device=device
                    )

                    done_buffer.append(done_tensor)

                    timeout_value_buffer.append(
                        timeout_values
                    )

                    hidden = (
                        next_hidden
                        * (1 - done_tensor.unsqueeze(-1))
                    )

                    previous_action = (
                        actions
                        * (1 - done_tensor.unsqueeze(-1))
                    )

                    global_step += N

                _, next_value, _ = model(
                    model_obs(
                        stack_obs(observations)
                    ),
                    hidden,
                    previous_action
                )

            # ==================================
            # PREPARE ROLLOUT TENSORS
            # ==================================

            obs_buffer = {
                key: torch.stack(values)
                for key, values in obs_buffer.items()
            }

            hidden_buffer = torch.stack(
                hidden_buffer
            )

            previous_buffer = torch.stack(
                previous_buffer
            )

            action_buffer = torch.stack(
                action_buffer
            )

            old_logprobs = torch.stack(
                logprob_buffer
            )

            values = torch.stack(
                value_buffer
            )

            rewards = torch.stack(
                reward_buffer
            )

            dones = torch.stack(
                done_buffer
            )

            timeout_values = torch.stack(
                timeout_value_buffer
            )

            # ==================================
            # GENERALIZED ADVANTAGE ESTIMATION
            # ==================================

            advantages = torch.zeros_like(
                rewards
            )

            gae = torch.zeros(
                N, device=device
            )

            for t in reversed(range(T)):

                following_value = (
                    next_value
                    if t == T - 1
                    else values[t + 1]
                )

                nonterminal = 1.0 - dones[t]

                delta = (
                    rewards[t]
                    + GAMMA * (
                        following_value * nonterminal
                        + timeout_values[t]
                    )
                    - values[t]
                )

                gae = (
                    delta
                    + GAMMA
                    * GAE_LAMBDA
                    * nonterminal
                    * gae
                )

                advantages[t] = gae

            returns = advantages + values

            advantages = (
                advantages - advantages.mean()
            ) / (
                advantages.std(unbiased=False)
                + 1e-8
            )

            # ==================================
            # RECURRENT PPO OPTIMIZATION
            # ==================================

            model.train()

            sequences = [
                (start, env_index)

                for start in range(0, T, SEQ)

                for env_index in range(N)
            ]

            losses = []
            policy_losses = []
            value_losses = []
            entropies = []

            for epoch in range(EPOCHS):

                np.random.shuffle(sequences)

                for batch_start in range(
                    0,
                    len(sequences),
                    BATCH_SEQUENCES
                ):

                    batch = sequences[
                        batch_start:
                        batch_start + BATCH_SEQUENCES
                    ]

                    starts = torch.tensor(
                        [item[0] for item in batch],
                        device=device
                    )

                    indices = torch.tensor(
                        [item[1] for item in batch],
                        device=device
                    )

                    recurrent_state = (
                        hidden_buffer[starts, indices]
                        .detach()
                    )

                    new_logprobs = []

                    new_values = []

                    new_entropies = []

                    batch_old_logprobs = []

                    batch_advantages = []

                    batch_returns = []

                    for k in range(SEQ):

                        times = starts + k

                        if k > 0:

                            reset_mask = (
                                1 - dones[times - 1, indices]
                            ).unsqueeze(-1)

                            recurrent_state = (
                                recurrent_state * reset_mask
                            )

                        data = {
                            key: value[times, indices]

                            for key, value in obs_buffer.items()
                        }

                        previous = previous_buffer[
                            times, indices
                        ]

                        logits, predicted_value, recurrent_state = model(
                            model_obs(data),
                            recurrent_state,
                            previous
                        )

                        distribution = (
                            torch.distributions.Bernoulli(
                                logits=logits
                            )
                        )

                        selected_actions = action_buffer[
                            times, indices
                        ]

                        new_logprobs.append(
                            distribution.log_prob(
                                selected_actions
                            ).sum(dim=-1)
                        )

                        new_entropies.append(
                            distribution.entropy()
                            .sum(dim=-1)
                        )

                        new_values.append(
                            predicted_value
                        )

                        batch_old_logprobs.append(
                            old_logprobs[times, indices]
                        )

                        batch_advantages.append(
                            advantages[times, indices]
                        )

                        batch_returns.append(
                            returns[times, indices]
                        )

                    new_logprobs = torch.stack(
                        new_logprobs
                    )

                    new_values = torch.stack(
                        new_values
                    )

                    old_lp = torch.stack(
                        batch_old_logprobs
                    )

                    adv = torch.stack(
                        batch_advantages
                    )

                    target_returns = torch.stack(
                        batch_returns
                    )

                    ratio = torch.exp(
                        new_logprobs - old_lp
                    )

                    unclipped = ratio * adv

                    clipped = torch.clamp(
                        ratio,
                        1 - CLIP,
                        1 + CLIP
                    ) * adv

                    policy_loss = -torch.minimum(
                        unclipped,
                        clipped
                    ).mean()

                    value_loss = F.mse_loss(
                        new_values,
                        target_returns
                    )

                    entropy = torch.stack(
                        new_entropies
                    ).mean()

                    loss = (
                        policy_loss
                        + VALUE_COEF * value_loss
                        - ENTROPY_COEF * entropy
                    )

                    if not torch.isfinite(loss):

                        raise RuntimeError(
                            "Non-finite PPO loss"
                        )

                    optimizer.zero_grad(
                        set_to_none=True
                    )

                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        0.5,
                        error_if_nonfinite=True
                    )

                    optimizer.step()

                    losses.append(loss.item())

                    policy_losses.append(
                        policy_loss.item()
                    )

                    value_losses.append(
                        value_loss.item()
                    )

                    entropies.append(
                        entropy.item()
                    )

            # ==================================
            # TRAINING METRICS
            # ==================================

            elapsed = (
                time.perf_counter() - start_time
            )

            fps = (
                global_step - initial_step
            ) / max(elapsed, 1e-8)

            def mean(data):

                return (
                    float(np.mean(data))
                    if len(data)
                    else 0.0
                )

            metrics = {
                "steps": global_step,
                "stage": "navigation_focus",
                "episodes": episodes,
                "fps": round(fps, 2),
                "reward": mean(recent_rewards),
                "kills": mean(recent_kills),
                "exit_rate": mean(recent_exits),
                "keys": mean(recent_keys),
                "cells": mean(recent_cells),
                "doors": mean(recent_doors),
                "loss": mean(losses),
                "policy_loss": mean(policy_losses),
                "value_loss": mean(value_losses),
                "entropy": mean(entropies)
            }

            print(
                f"Update {update:4d} | "
                f"Steps {global_step:8d} | "
                f"Objective focus | "
                f"FPS {fps:6.1f} | "
                f"Reward {metrics['reward']:+7.3f} | "
                f"Kills {metrics['kills']:.2f} | "
                f"Exits {metrics['exit_rate']:.1%} | "
                f"Keys {metrics['keys']:.2f} | "
                f"Cells {metrics['cells']:.1f} | "
                f"Loss {metrics['loss']:.4f}",
                flush=True
            )

            write_header = not LOG_FILE.exists()

            with LOG_FILE.open(
                "a", newline=""
            ) as file:

                writer = csv.DictWriter(
                    file,
                    fieldnames=list(metrics)
                )

                if write_header:

                    writer.writeheader()

                writer.writerow(metrics)

            # Save the first successful update,
            # then every 10,000 decisions.

            if (
                update == 1
                or global_step - last_checkpoint >= 10000
            ):

                save_checkpoint()

                last_checkpoint = global_step

                print(
                    f"Checkpoint saved: {global_step}",
                    flush=True
                )

        save_checkpoint()

        print("\nTRAINING COMPLETE")

        print("Total decisions:", global_step)

        print("Completed episodes:", episodes)

        print("Weights:", CHECKPOINT)

    except KeyboardInterrupt:

        # Keep the last fully optimized checkpoint, not a partial rollout.

        print("\nTraining interrupted.")

        print("Checkpoint saved:", CHECKPOINT)

    finally:

        for env in envs:

            env.close()


if __name__ == "__main__":
    mp.freeze_support()
    main()
