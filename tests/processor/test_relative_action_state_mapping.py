#!/usr/bin/env python

import torch

from lerobot.processor import TransitionKey, batch_to_transition, create_transition
from lerobot.processor.relative_action_processor import (
    AbsoluteActionsProcessorStep,
    RelativeActionsProcessorStep,
)
from lerobot.utils.constants import ACTION, OBS_STATE

REFERENCE_INDICES = [0, 1, 2, 3, 4, 5, 6, 21, 22, 23, 24, 25, 26, 27]


def test_relative_actions_use_latest_temporal_state_and_configured_dimensions():
    state = torch.arange(2 * 3 * 42, dtype=torch.float32).reshape(2, 3, 42)
    actions = torch.randn(2, 5, 14)
    step = RelativeActionsProcessorStep(
        enabled=True,
        reference_state_index=-1,
        reference_state_indices=REFERENCE_INDICES,
    )

    result = step(batch_to_transition({OBS_STATE: state, ACTION: actions}))

    anchor = state[:, -1, REFERENCE_INDICES]
    torch.testing.assert_close(result[TransitionKey.ACTION], actions - anchor.unsqueeze(1))
    torch.testing.assert_close(step.get_cached_state(), anchor)


def test_indexed_relative_postprocessor_returns_absolute_actions_at_inference():
    state = torch.arange(42, dtype=torch.float32).unsqueeze(0)
    relative_actions = torch.randn(1, 4, 14)
    relative_step = RelativeActionsProcessorStep(
        enabled=True,
        reference_state_index=-1,
        reference_state_indices=REFERENCE_INDICES,
    )
    absolute_step = AbsoluteActionsProcessorStep(enabled=True, relative_step=relative_step)

    # Inference preprocessing has observations but no target action. It must still
    # cache the correctly mapped anchor for output postprocessing.
    relative_step(batch_to_transition({OBS_STATE: state}))
    result = absolute_step(create_transition(action=relative_actions))

    anchor = state[:, REFERENCE_INDICES]
    torch.testing.assert_close(result[TransitionKey.ACTION], relative_actions + anchor.unsqueeze(1))


def test_relative_state_mapping_is_serialized():
    step = RelativeActionsProcessorStep(
        enabled=True,
        reference_state_index=-1,
        reference_state_indices=REFERENCE_INDICES,
    )

    config = step.get_config()

    assert config["reference_state_index"] == -1
    assert config["reference_state_indices"] == REFERENCE_INDICES
