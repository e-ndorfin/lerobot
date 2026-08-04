#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
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
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    ObservationProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import OBS_STATE, POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_act import ACTConfig


@dataclass
@ProcessorStepRegistry.register(name="select_robot_state_processor")
class SelectRobotStateProcessorStep(ObservationProcessorStep):
    """Select fixed elements from the final dimension of ``observation.state``."""

    indices: list[int]

    def __post_init__(self) -> None:
        if not self.indices:
            raise ValueError("indices must not be empty")
        if any(not isinstance(index, int) or isinstance(index, bool) for index in self.indices):
            raise ValueError(f"indices must contain only integers: {self.indices}")
        if any(index < 0 for index in self.indices):
            raise ValueError(f"indices must be non-negative: {self.indices}")
        if len(self.indices) != len(set(self.indices)):
            raise ValueError(f"indices contains duplicates: {self.indices}")

    def observation(self, observation):
        if OBS_STATE not in observation:
            raise ValueError("State selection requires observation.state")

        state = observation[OBS_STATE]
        if not hasattr(state, "shape"):
            state = np.asarray(state)
        if state.ndim < 1 or state.shape[-1] <= max(self.indices):
            raise ValueError(
                f"Cannot select state indices {self.indices} from observation.state shape {tuple(state.shape)}"
            )
        observation[OBS_STATE] = state[..., self.indices]
        return observation

    def get_config(self) -> dict[str, Any]:
        return {"indices": self.indices}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        transformed = deepcopy(features)
        observation_features = transformed.get(PipelineFeatureType.OBSERVATION, {})
        if OBS_STATE in observation_features:
            feature = observation_features[OBS_STATE]
            observation_features[OBS_STATE] = PolicyFeature(type=feature.type, shape=(len(self.indices),))
        return transformed


def _select_robot_state_stats(
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None,
    indices: list[int] | None,
) -> dict[str, dict[str, torch.Tensor]] | None:
    """Return normalization stats matching the selected robot-state elements."""
    if dataset_stats is None or indices is None or OBS_STATE not in dataset_stats:
        return dataset_stats

    selected_stats = dict(dataset_stats)
    selected_state_stats = {}
    for name, value in dataset_stats[OBS_STATE].items():
        if hasattr(value, "shape") and len(value.shape) > 0 and value.shape[0] > max(indices):
            selected_state_stats[name] = value[indices]
        elif isinstance(value, list) and len(value) > max(indices):
            selected_state_stats[name] = [value[index] for index in indices]
        else:
            selected_state_stats[name] = value
    selected_stats[OBS_STATE] = selected_state_stats
    return selected_stats


def make_act_pre_post_processors(
    config: ACTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Creates the pre- and post-processing pipelines for the ACT policy.

    The pre-processing pipeline handles normalization, batching, and device placement for the model inputs.
    The post-processing pipeline handles unnormalization and moves the model outputs back to the CPU.

    Args:
        config (ACTConfig): The ACT policy configuration object.
        dataset_stats (dict[str, dict[str, torch.Tensor]] | None): A dictionary containing dataset
            statistics (e.g., mean and std) used for normalization. Defaults to None.

    Returns:
        tuple[PolicyProcessorPipeline[dict[str, Any], dict[str, Any]], PolicyProcessorPipeline[PolicyAction, PolicyAction]]: A tuple containing the
        pre-processor pipeline and the post-processor pipeline.
    """

    input_steps = [RenameObservationsProcessorStep(rename_map={})]
    if config.robot_state_indices is not None:
        input_steps.append(SelectRobotStateProcessorStep(indices=config.robot_state_indices))
        dataset_stats = _select_robot_state_stats(dataset_stats, config.robot_state_indices)
    input_steps.extend(
        [
            AddBatchDimensionProcessorStep(),
            DeviceProcessorStep(device=config.device),
            NormalizerProcessorStep(
                features={**config.input_features, **config.output_features},
                norm_map=config.normalization_mapping,
                stats=dataset_stats,
                device=config.device,
            ),
        ]
    )
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
