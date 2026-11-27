
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoomCortex(nn.Module):

    def __init__(self):
        super().__init__()

        self.hidden_size = 256

        # 360-degree spatial ray encoder

        self.ray_encoder = nn.Sequential(
            nn.Conv1d(
                7, 32, kernel_size=5,
                padding=2,
                padding_mode="circular"
            ),
            nn.ReLU(),

            nn.Conv1d(
                32, 64, kernel_size=5,
                stride=2,
                padding=2,
                padding_mode="circular"
            ),
            nn.ReLU(),

            nn.Conv1d(
                64, 64, kernel_size=3,
                stride=2,
                padding=1,
                padding_mode="circular"
            ),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(64 * 16, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        # Entity type embeddings

        self.entity_embedding = nn.Embedding(
            12, 16, padding_idx=0
        )

        # Individual entity processing

        self.entity_encoder = nn.Sequential(
            nn.Linear(21, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Player vitals

        self.vitals_encoder = nn.Sequential(
            nn.Linear(17, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.register_buffer(
            "vitals_scale",
            torch.tensor([
                200, 200, 7, 300,
                400, 100, 100, 600,
                1, 1, 1, 1, 1, 1,
                20, 20, 20
            ], dtype=torch.float32)
        )

        # Context-aware entity attention

        self.entity_query = nn.Linear(
            128 + 64, 64
        )

        self.entity_key = nn.Linear(
            64, 64
        )

        self.entity_value = nn.Linear(
            64, 64
        )

        # Feature fusion

        self.fusion = nn.Sequential(
            nn.Linear(128 + 64 + 64 + 9, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )

        # Recurrent spatial memory

        self.memory = nn.GRUCell(
            input_size=256,
            hidden_size=256
        )

        # Actor: nine binary action logits

        self.actor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 9)
        )

        # Critic: expected future reward

        self.critic = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.initialize_weights()

    def initialize_weights(self):

        for module in self.modules():

            if isinstance(module, nn.Linear):

                nn.init.orthogonal_(
                    module.weight,
                    gain=2 ** 0.5
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.orthogonal_(
            self.actor[-1].weight,
            gain=0.01
        )

        nn.init.orthogonal_(
            self.critic[-1].weight,
            gain=1.0
        )

    def initial_state(self, batch_size=1):

        device = next(
            self.parameters()
        ).device

        return torch.zeros(
            batch_size,
            self.hidden_size,
            device=device
        )

    def encode_rays(self, observation):

        distances = observation[
            "rays"
        ]["distance"].float()

        hit_types = observation[
            "rays"
        ]["hit_type"].long()

        hit_types = F.one_hot(
            hit_types,
            num_classes=6
        ).float()

        ray_input = torch.cat(
            [
                distances.unsqueeze(-1),
                hit_types
            ],
            dim=-1
        )

        ray_input = ray_input.transpose(
            1, 2
        )

        return self.ray_encoder(ray_input)

    def encode_vitals(self, observation):

        vitals = observation[
            "vitals"
        ].float()

        vitals = torch.nan_to_num(vitals)

        vitals = (
            vitals / self.vitals_scale
        ).clamp(-5.0, 5.0)

        return self.vitals_encoder(vitals)

    def encode_entities(
        self,
        observation,
        ray_features,
        vital_features
    ):

        entities = observation[
            "visible_entities"
        ]

        features = torch.nan_to_num(
            entities["features"].float()
        )

        features = torch.cat(
            [
                features[..., :3],
                features[..., 3:] / 20.0
            ],
            dim=-1
        ).clamp(-5.0, 5.0)

        types = entities[
            "entity_type"
        ].long()

        mask = entities[
            "mask"
        ].bool()

        embeddings = self.entity_embedding(
            types
        )

        entity_input = torch.cat(
            [features, embeddings],
            dim=-1
        )

        encoded = self.entity_encoder(
            entity_input
        )

        context = torch.cat(
            [ray_features, vital_features],
            dim=-1
        )

        query = self.entity_query(
            context
        ).unsqueeze(1)

        keys = self.entity_key(encoded)

        values = self.entity_value(encoded)

        scores = (
            query * keys
        ).sum(dim=-1) / 8.0

        scores = scores.masked_fill(
            ~mask, -1e9
        )

        attention = torch.softmax(
            scores,
            dim=-1
        )

        attention = (
            attention * mask.float()
        )

        attention = attention / (
            attention.sum(
                dim=-1,
                keepdim=True
            ).clamp_min(1e-8)
        )

        pooled = (
            attention.unsqueeze(-1)
            * values
        ).sum(dim=1)

        return pooled

    def forward(
        self,
        observation,
        hidden,
        previous_action=None
    ):

        ray_features = self.encode_rays(
            observation
        )

        vital_features = self.encode_vitals(
            observation
        )

        entity_features = self.encode_entities(
            observation,
            ray_features,
            vital_features
        )

        if previous_action is None:

            previous_action = torch.zeros(
                ray_features.shape[0],
                9,
                device=ray_features.device
            )

        fused = self.fusion(
            torch.cat(
                [
                    ray_features,
                    entity_features,
                    vital_features,
                    previous_action.float()
                ],
                dim=-1
            )
        )

        hidden = self.memory(
            fused, hidden
        )

        logits = self.actor(hidden)

        value = self.critic(
            hidden
        ).squeeze(-1)

        return logits, value, hidden
