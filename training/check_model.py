
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

from doom_env import make_env
from submission.model import DoomCortex


torch.set_num_threads(1)

device = torch.device("cpu")


def to_tensor(observation):

    if isinstance(observation, dict):

        return {
            key: to_tensor(value)
            for key, value in observation.items()
        }

    return torch.as_tensor(
        observation,
        device=device
    ).unsqueeze(0)


env = make_env(
    str(ROOT / "maps" / "two_rooms.wad"),
    "MAP01",
    episode_ticks=2100,
    render_buffers=False
)

try:

    obs, info = env.reset(seed=42)

    model = DoomCortex().to(device)

    model.eval()

    hidden = model.initial_state()

    previous_action = torch.zeros(
        1, 9, device=device
    )

    tensor_obs = to_tensor(obs)

    print("\nPROJECT-DOOM MODEL VALIDATION")
    print("=" * 60)

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    memory_bytes = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    )

    print("Trainable parameters:", parameters)

    print(
        f"Parameter memory: "
        f"{memory_bytes / 1024**2:.3f} MB"
    )

    with torch.inference_mode():

        logits, value, new_hidden = model(
            tensor_obs,
            hidden,
            previous_action
        )

    print("\nOUTPUT DIMENSIONS")
    print("=" * 60)

    print("Action logits:", logits.shape)
    print("Critic value:", value.shape)
    print("Recurrent memory:", new_hidden.shape)

    assert logits.shape == (1, 9)
    assert value.shape == (1,)
    assert new_hidden.shape == (1, 256)

    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()
    assert torch.isfinite(new_hidden).all()

    print("All outputs finite: True")

    print("\nEMPTY ENTITY TEST")
    print("=" * 60)

    empty_obs = to_tensor(obs)

    empty_obs["visible_entities"]["mask"] = (
        torch.zeros_like(
            empty_obs["visible_entities"]["mask"]
        )
    )

    with torch.inference_mode():

        empty_logits, _, _ = model(
            empty_obs,
            hidden,
            previous_action
        )

    assert torch.isfinite(empty_logits).all()

    print("Empty entity handling: Passed")

    print("\nLIVE ENVIRONMENT TEST")
    print("=" * 60)

    hidden = model.initial_state()

    latencies = []

    total_reward = 0.0

    with torch.inference_mode():

        for step in range(100):

            tensor_obs = to_tensor(obs)

            start = time.perf_counter()

            logits, value, hidden = model(
                tensor_obs,
                hidden,
                previous_action
            )

            distribution = (
                torch.distributions.Bernoulli(
                    logits=logits
                )
            )

            action_tensor = distribution.sample()

            action = (
                action_tensor
                .squeeze(0)
                .cpu()
                .numpy()
                .astype(np.int8)
            )

            latencies.append(
                (time.perf_counter() - start)
                * 1000
            )

            assert env.action_space.contains(
                action
            )

            obs, reward, terminated, truncated, info = (
                env.step(action)
            )

            previous_action = action_tensor

            total_reward += reward

            if terminated or truncated:

                print("Episode finished.")

                obs, info = env.reset()

                hidden = model.initial_state()

                previous_action.zero_()

    print("Valid decisions:", len(latencies))

    print(
        f"Mean latency: "
        f"{np.mean(latencies):.3f} ms"
    )

    print(
        f"P95 latency: "
        f"{np.percentile(latencies, 95):.3f} ms"
    )

    print(
        f"Maximum latency: "
        f"{np.max(latencies):.3f} ms"
    )

    print(
        f"Total reward: {total_reward:.3f}"
    )

    print("\nGRADIENT FLOW TEST")
    print("=" * 60)

    model.train()
    model.zero_grad(set_to_none=True)

    logits, value, hidden = model(
        to_tensor(obs),
        model.initial_state()
    )

    loss = (
        logits.square().mean()
        + value.square().mean()
    )

    loss.backward()

    gradients_valid = all(
        torch.isfinite(p.grad).all().item()
        for p in model.parameters()
        if p.grad is not None
    )

    assert gradients_valid

    print("Finite gradients:", gradients_valid)

    print("\nALL MODEL TESTS PASSED")

finally:

    env.close()
