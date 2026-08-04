#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.diffusion.modeling_diffusion import DiffusionModel
from lerobot.policies.multi_task_dit.modeling_multi_task_dit import (
    DiffusionObjective,
    FlowMatchingObjective,
)
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE


class FixedACTModel(nn.Module):
    def __init__(self, actions: torch.Tensor, mu: torch.Tensor, log_variance: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("actions", actions)
        self.register_buffer("mu", mu)
        self.register_buffer("log_variance", log_variance)

    def forward(self, batch):
        return self.actions, (self.mu, self.log_variance)


class ZeroDenoiser(nn.Module):
    def forward(self, trajectory, timesteps, global_cond):
        return torch.zeros_like(trajectory)


class ZeroConditionedDenoiser(nn.Module):
    def forward(self, trajectory, timesteps, conditioning_vec):
        return torch.zeros_like(trajectory)


class NoisingScheduler:
    config = SimpleNamespace(num_train_timesteps=10, prediction_type="sample")

    def add_noise(self, trajectory, noise, timesteps):
        return trajectory


def test_act_unreduced_loss_uses_each_samples_valid_actions_and_kl():
    policy = object.__new__(ACTPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(image_features=[], use_vae=True, kl_weight=0.1)
    policy.model = FixedACTModel(
        actions=torch.tensor(
            [
                [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
                [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]],
            ]
        ),
        mu=torch.tensor([[1.0, 0.0], [2.0, 0.0]]),
        log_variance=torch.zeros(2, 2),
    )
    batch = {
        ACTION: torch.zeros(2, 3, 2),
        "action_is_pad": torch.tensor([[False, True, True], [False, False, False]]),
    }

    per_sample_loss, _ = policy.forward(batch, reduction="none")
    mean_loss, _ = policy.forward(batch)

    torch.testing.assert_close(per_sample_loss, torch.tensor([1.05, 2.2]))
    assert mean_loss.item() == pytest.approx(1.875)


def test_diffusion_unreduced_loss_uses_each_samples_valid_actions():
    model = object.__new__(DiffusionModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        horizon=3,
        n_obs_steps=1,
        prediction_type="sample",
        do_mask_loss_for_padding=True,
    )
    model.noise_scheduler = NoisingScheduler()
    model.unet = ZeroDenoiser()
    model._prepare_global_conditioning = lambda batch: torch.zeros(len(batch[ACTION]), 1)
    batch = {
        OBS_STATE: torch.zeros(2, 1, 2),
        OBS_ENV_STATE: torch.zeros(2, 1, 2),
        ACTION: torch.tensor(
            [
                [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
                [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]],
            ]
        ),
        "action_is_pad": torch.tensor([[False, True, True], [False, False, False]]),
    }

    per_sample_loss = model.compute_loss(batch, reduction="none")
    mean_loss = model.compute_loss(batch)

    torch.testing.assert_close(per_sample_loss, torch.tensor([1.0, 4.0]))
    assert mean_loss.item() == pytest.approx(3.25)


@pytest.mark.parametrize("objective_class", [DiffusionObjective, FlowMatchingObjective])
def test_multi_task_dit_objectives_return_one_masked_loss_per_sample(monkeypatch, objective_class):
    objective = object.__new__(objective_class)
    nn.Module.__init__(objective)
    objective.config = SimpleNamespace(prediction_type="sample", sigma_min=0.0)
    objective.do_mask_loss_for_padding = True
    if objective_class is DiffusionObjective:
        objective.noise_scheduler = NoisingScheduler()
    else:
        objective._sample_timesteps = lambda batch_size, device: torch.ones(batch_size, device=device)

    monkeypatch.setattr(torch, "randn_like", torch.zeros_like)
    batch = {
        ACTION: torch.tensor(
            [
                [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
                [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]],
            ]
        ),
        "action_is_pad": torch.tensor([[False, True, True], [False, False, False]]),
    }

    per_sample_loss = objective.compute_loss(
        ZeroConditionedDenoiser(),
        batch,
        conditioning_vec=torch.zeros(2, 1),
        reduction="none",
    )

    torch.testing.assert_close(per_sample_loss, torch.tensor([1.0, 4.0]))
