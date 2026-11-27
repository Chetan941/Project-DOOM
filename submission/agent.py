from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

SUBMISSION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SUBMISSION_DIR))

from model import DoomCortex


class DoomAgent:
    def __init__(self, observation_space, action_space, device: str):
        self.device = torch.device(device)
        torch.set_num_threads(1)

        self.model = DoomCortex().to(self.device)

        checkpoint = torch.load(
            SUBMISSION_DIR / "weights" / "doom_cortex_keys.pt",
            map_location="cpu",
            weights_only=True
        )

        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

        self.reset()

    def reset(self):
        self.hidden = self.model.initial_state(1)

        self.previous_action = torch.zeros(
            (1, 9),
            dtype=torch.float32,
            device=self.device
        )

    def tensor(self, value):
        return torch.as_tensor(
            value, device=self.device
        ).unsqueeze(0)

    @torch.inference_mode()
    def act(self, observation):
        obs = {
            "rays": {
                "distance": self.tensor(
                    observation["rays"]["distance"]
                ),
                "hit_type": self.tensor(
                    observation["rays"]["hit_type"]
                )
            },
            "vitals": self.tensor(observation["vitals"]),
            "visible_entities": {
                "features": self.tensor(
                    observation["visible_entities"]["features"]
                ),
                "entity_type": self.tensor(
                    observation["visible_entities"]["entity_type"]
                ),
                "mask": self.tensor(
                    observation["visible_entities"]["mask"]
                )
            }
        }

        logits, _, self.hidden = self.model(
            obs,
            self.hidden,
            self.previous_action
        )

        action = torch.bernoulli(torch.sigmoid(logits))

        self.previous_action = action

        return action.squeeze(0).cpu().numpy().astype(np.int8)