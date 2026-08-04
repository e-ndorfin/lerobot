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

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from lerobot.scripts.lerobot_train import (
    _forward_grouped_by_weight,
    _supports_unreduced_loss,
    update_policy,
)


class ScalarLossPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.seen_batches: list[list[float]] = []

    def forward(self, batch):
        self.seen_batches.append(batch["value"].detach().cpu().tolist())
        per_sample_loss = self.scale * batch["value"]
        return per_sample_loss.mean(), {"loss": per_sample_loss.mean().item()}


class UnreducedLossPolicy(ScalarLossPolicy):
    def forward(self, batch, reduction="mean"):
        per_sample_loss = self.scale * batch["value"]
        if reduction == "none":
            return per_sample_loss, {"loss": per_sample_loss.mean().item()}
        return per_sample_loss.mean(), {"loss": per_sample_loss.mean().item()}


class FakeAccelerator:
    def autocast(self):
        return nullcontext()

    def backward(self, loss):
        loss.backward()

    def clip_grad_norm_(self, parameters, max_norm):
        return nn.utils.clip_grad_norm_(parameters, max_norm)

    def reduce(self, value, reduction):
        assert reduction == "sum"
        return value

    def unwrap_model(self, policy, keep_fp32_wrapper=True):
        assert keep_fp32_wrapper
        return policy


def test_grouped_fallback_matches_weighted_per_sample_objective():
    policy = ScalarLossPolicy()
    batch = {
        "value": torch.tensor([1.0, 2.0, 4.0, 8.0]),
        "task": ["a", "b", "c", "d"],
        "constant": torch.tensor(3.0),
    }
    weights = torch.tensor([1.0, 1.0, 2.0, 0.0])

    loss, output = _forward_grouped_by_weight(policy, batch, weights)

    expected = (1.0 + 2.0 + 2.0 * 4.0) / weights.sum()
    assert loss.item() == pytest.approx(expected.item())
    assert policy.seen_batches == [[1.0, 2.0], [4.0]]
    assert output["sample_weight_grouped_forwards"] == 2


def test_grouped_fallback_returns_differentiable_zero_for_all_zero_weights():
    policy = ScalarLossPolicy()

    loss, output = _forward_grouped_by_weight(
        policy,
        {"value": torch.tensor([1.0, 2.0])},
        torch.zeros(2),
    )
    loss.backward()

    assert loss.item() == 0.0
    assert policy.scale.grad.item() == 0.0
    assert output["sample_weight_grouped_forwards"] == 1


def test_unreduced_loss_support_detection_requires_explicit_argument():
    assert not _supports_unreduced_loss(ScalarLossPolicy())
    assert _supports_unreduced_loss(UnreducedLossPolicy())


def test_zero_weight_batch_skips_optimizer_weight_decay_and_scheduler():
    policy = ScalarLossPolicy()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.1, weight_decay=0.5)
    scheduler = Mock()
    initial_scale = policy.scale.detach().clone()

    _, output = update_policy(
        train_metrics=SimpleNamespace(),
        policy=policy,
        batch={"value": torch.tensor([1.0, 2.0])},
        optimizer=optimizer,
        grad_clip_norm=1.0,
        accelerator=FakeAccelerator(),
        lr_scheduler=scheduler,
        sample_weights=torch.zeros(2),
        supports_unreduced_loss=False,
    )

    torch.testing.assert_close(policy.scale, initial_scale)
    scheduler.step.assert_not_called()
    assert output["sample_weight_skipped_update"] == 1
