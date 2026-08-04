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

"""Tests for the sample weighting infrastructure."""

from unittest.mock import Mock

import draccus
import pytest

pytest.importorskip("pandas", reason="pandas is required (install lerobot[dataset])")

import torch

from lerobot.utils.sample_weighting import (
    CONTROL_MODE_KEY,
    ControlModeWeighter,
    SampleWeighter,
    SampleWeightingConfig,
    UniformWeighter,
    make_sample_weighter,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_progress_parquet(tmp_path):
    """Create a sample progress parquet file for testing."""
    import pandas as pd

    # Create sample progress data for 2 episodes with 10 frames each
    data = {
        "index": list(range(20)),
        "episode_index": [0] * 10 + [1] * 10,
        "frame_index": list(range(10)) * 2,
        "progress_sparse": [i / 10.0 for i in range(10)] * 2,
    }
    df = pd.DataFrame(data)
    parquet_path = tmp_path / "sarm_progress.parquet"
    df.to_parquet(parquet_path)
    return parquet_path


# =============================================================================
# SampleWeightingConfig Tests
# =============================================================================


def test_config_default_values():
    """Test default configuration values."""
    config = SampleWeightingConfig()
    assert config.type == "rabc"
    assert config.progress_path is None
    assert config.head_mode == "sparse"
    assert config.kappa == 0.01
    assert config.epsilon == 1e-6
    assert config.mode_weights == {}
    assert config.control_mode_key == CONTROL_MODE_KEY
    assert config.default_weight is None
    assert config.extra_params == {}


def test_config_custom_values():
    """Test configuration with custom values."""
    config = SampleWeightingConfig(
        type="rabc",
        progress_path="/path/to/progress.parquet",
        head_mode="dense",
        kappa=0.05,
        epsilon=1e-8,
        extra_params={"fallback_weight": 0.5},
    )
    assert config.type == "rabc"
    assert config.progress_path == "/path/to/progress.parquet"
    assert config.head_mode == "dense"
    assert config.kappa == 0.05
    assert config.epsilon == 1e-8
    assert config.extra_params == {"fallback_weight": 0.5}


def test_config_uniform_type():
    """Test configuration for uniform weighting."""
    config = SampleWeightingConfig(type="uniform")
    assert config.type == "uniform"


def test_config_control_mode_type():
    config = SampleWeightingConfig(type="control_mode", mode_weights={0: 1.0, 2: 3.0, 4: 0.0})

    assert config.type == "control_mode"
    assert config.mode_weights == {0: 1.0, 2: 3.0, 4: 0.0}


def test_config_control_mode_decodes_stringified_cli_mapping_keys():
    config = draccus.decode(
        SampleWeightingConfig,
        {"type": "control_mode", "mode_weights": {"0": 1.0, "2": 3.0}},
    )

    assert config.mode_weights == {0: 1.0, 2: 3.0}


# =============================================================================
# UniformWeighter Tests
# =============================================================================


def test_uniform_weighter_inherits_from_sample_weighter():
    """Test that UniformWeighter is a SampleWeighter."""
    weighter = UniformWeighter(device=torch.device("cpu"))
    assert isinstance(weighter, SampleWeighter)


def test_uniform_weighter_compute_batch_weights_with_action_key():
    """Test weight computation with 'action' key in batch."""
    weighter = UniformWeighter(device=torch.device("cpu"))
    batch = {"action": torch.randn(8, 10)}

    weights, stats = weighter.compute_batch_weights(batch)

    assert weights.shape == (8,)
    assert torch.allclose(weights, torch.ones(8))
    assert stats["mean_weight"] == 1.0
    assert stats["type"] == "uniform"


def test_uniform_weighter_compute_batch_weights_with_index_key():
    """Test weight computation with 'index' key in batch."""
    weighter = UniformWeighter(device=torch.device("cpu"))
    batch = {"index": torch.arange(16)}

    weights, stats = weighter.compute_batch_weights(batch)

    assert weights.shape == (16,)
    assert torch.allclose(weights, torch.ones(16))


def test_uniform_weighter_compute_batch_weights_no_tensor_keys():
    """Test weight computation with no tensor keys (fallback to size 1)."""
    weighter = UniformWeighter(device=torch.device("cpu"))
    batch = {"other_key": "some_value"}

    weights, stats = weighter.compute_batch_weights(batch)

    assert weights.shape == (1,)
    assert torch.allclose(weights, torch.ones(1))


def test_uniform_weighter_compute_batch_weights_empty_batch_raises():
    """Test that empty batch raises ValueError."""
    weighter = UniformWeighter(device=torch.device("cpu"))
    batch = {}

    with pytest.raises(ValueError, match="empty batch"):
        weighter.compute_batch_weights(batch)


def test_uniform_weighter_compute_batch_weights_scans_all_keys():
    """Test that batch size is determined by scanning all tensor values."""
    weighter = UniformWeighter(device=torch.device("cpu"))
    # Batch with non-standard key containing a tensor
    batch = {"custom_tensor": torch.randn(7, 3)}

    weights, stats = weighter.compute_batch_weights(batch)

    assert weights.shape == (7,)
    assert torch.allclose(weights, torch.ones(7))


def test_uniform_weighter_compute_batch_weights_on_cuda():
    """Test that weights are placed on the correct device."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    weighter = UniformWeighter(device=torch.device("cuda"))
    batch = {"action": torch.randn(4, 10)}

    weights, _ = weighter.compute_batch_weights(batch)

    assert weights.device.type == "cuda"


def test_uniform_weighter_get_stats():
    """Test get_stats returns expected structure."""
    weighter = UniformWeighter(device=torch.device("cpu"))
    stats = weighter.get_stats()

    assert stats == {"type": "uniform"}


# =============================================================================
# ControlModeWeighter Tests
# =============================================================================


def test_control_mode_weighter_normalizes_configured_ratios():
    weighter = ControlModeWeighter(
        mode_weights={0: 1.0, 1: 0.5, 2: 2.0, 4: 0.0},
        device=torch.device("cpu"),
    )
    batch = {CONTROL_MODE_KEY: torch.tensor([0.0, 1.0, 2.0, 4.0])}

    weights, stats = weighter.compute_batch_weights(batch)

    expected = torch.tensor([1.0, 0.5, 2.0, 0.0])
    expected *= len(expected) / expected.sum()
    assert torch.allclose(weights, expected)
    assert weights.sum().item() == pytest.approx(4.0)
    assert stats["mode_0_count"] == 1
    assert stats["mode_4_count"] == 1
    assert stats["zero_weight_batch"] == 0


def test_control_mode_weighter_accepts_singleton_feature_dimension():
    weighter = ControlModeWeighter(mode_weights={0: 1.0, 2: 2.0}, device=torch.device("cpu"))

    weights, _ = weighter.compute_batch_weights({CONTROL_MODE_KEY: torch.tensor([[0.0], [2.0]])})

    assert weights.shape == (2,)
    assert weights.sum().item() == pytest.approx(2.0)


def test_control_mode_weighter_uses_explicit_default_weight():
    weighter = ControlModeWeighter(
        mode_weights={0: 1.0},
        default_weight=0.25,
        device=torch.device("cpu"),
    )

    weights, _ = weighter.compute_batch_weights({CONTROL_MODE_KEY: torch.tensor([0.0, 9.0])})

    assert torch.allclose(weights, torch.tensor([1.6, 0.4]))


def test_control_mode_weighter_rejects_unknown_mode_without_default():
    weighter = ControlModeWeighter(mode_weights={0: 1.0}, device=torch.device("cpu"))

    with pytest.raises(ValueError, match="No BC weight configured"):
        weighter.compute_batch_weights({CONTROL_MODE_KEY: torch.tensor([0.0, 2.0])})


@pytest.mark.parametrize(
    "batch,error",
    [
        ({}, "requires raw batch feature"),
        ({CONTROL_MODE_KEY: torch.tensor([0.5])}, "non-integer labels"),
        ({CONTROL_MODE_KEY: torch.tensor([[0.0, 1.0]])}, "one scalar label per sample"),
    ],
)
def test_control_mode_weighter_rejects_invalid_batch(batch, error):
    weighter = ControlModeWeighter(mode_weights={0: 1.0}, device=torch.device("cpu"))

    with pytest.raises(ValueError, match=error):
        weighter.compute_batch_weights(batch)


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"mode_weights": {}}, "non-empty"),
        ({"mode_weights": {0: -1.0}}, "non-negative"),
        ({"mode_weights": {0: float("inf")}}, "finite"),
        ({"mode_weights": {"0": 1.0}}, "labels must be integers"),
    ],
)
def test_control_mode_weighter_rejects_invalid_config(kwargs, error):
    with pytest.raises(ValueError, match=error):
        ControlModeWeighter(device=torch.device("cpu"), **kwargs)


def test_control_mode_weighter_preserves_all_zero_batch():
    weighter = ControlModeWeighter(mode_weights={4: 0.0}, device=torch.device("cpu"))

    weights, stats = weighter.compute_batch_weights({CONTROL_MODE_KEY: torch.tensor([4.0, 4.0])})

    assert torch.equal(weights, torch.zeros(2))
    assert stats["zero_weight_batch"] == 1
    assert weighter.get_stats()["zero_weight_batches"] == 1


# =============================================================================
# make_sample_weighter Factory Tests
# =============================================================================


def test_factory_returns_none_for_none_config():
    """Test that None config returns None weighter."""
    policy = Mock()
    device = torch.device("cpu")

    result = make_sample_weighter(None, policy, device)

    assert result is None


def test_factory_creates_uniform_weighter():
    """Test creation of UniformWeighter."""
    config = SampleWeightingConfig(type="uniform")
    policy = Mock()
    device = torch.device("cpu")

    weighter = make_sample_weighter(config, policy, device)

    assert isinstance(weighter, UniformWeighter)
    assert isinstance(weighter, SampleWeighter)


def test_factory_creates_control_mode_weighter():
    config = SampleWeightingConfig(type="control_mode", mode_weights={0: 1.0, 2: 2.0})
    policy = Mock()

    weighter = make_sample_weighter(config, policy, torch.device("cpu"))

    assert isinstance(weighter, ControlModeWeighter)
    assert isinstance(weighter, SampleWeighter)


def test_factory_raises_for_unknown_type():
    """Test that unknown type raises ValueError."""
    config = SampleWeightingConfig(type="unknown_type")
    policy = Mock()
    device = torch.device("cpu")

    with pytest.raises(ValueError, match="Unknown sample weighting type"):
        make_sample_weighter(config, policy, device)


def test_factory_rabc_requires_chunk_size():
    """Test that RABC weighter requires chunk_size in policy config."""
    config = SampleWeightingConfig(
        type="rabc",
        progress_path="/path/to/progress.parquet",
    )
    policy = Mock()
    policy.config = Mock()
    policy.config.chunk_size = None  # No chunk_size
    device = torch.device("cpu")

    with pytest.raises(ValueError, match="chunk_size"):
        make_sample_weighter(config, policy, device)


def test_factory_rabc_requires_progress_path_or_dataset_info():
    """Test that RABC weighter requires progress_path or dataset info for auto-detection."""
    config = SampleWeightingConfig(
        type="rabc",
        progress_path=None,  # No progress path
    )
    policy = Mock()
    policy.config = Mock()
    policy.config.chunk_size = 50
    device = torch.device("cpu")

    # Should fail when no progress_path AND no dataset info
    with pytest.raises(ValueError, match="progress_path"):
        make_sample_weighter(config, policy, device)


def test_factory_rabc_auto_detects_from_dataset_root(sample_progress_parquet):
    """Test that RABC weighter auto-detects progress_path from dataset_root."""
    config = SampleWeightingConfig(
        type="rabc",
        progress_path=None,  # Not provided, should auto-detect
    )
    policy = Mock()
    policy.config = Mock()
    policy.config.chunk_size = 5
    device = torch.device("cpu")

    # The parquet file is at sample_progress_parquet, get its parent directory
    dataset_root = sample_progress_parquet.parent
    weighter = make_sample_weighter(
        config,
        policy,
        device,
        dataset_root=str(dataset_root),
    )

    assert weighter is not None
    from lerobot.rewards.sarm.rabc import RABCWeights

    assert isinstance(weighter, RABCWeights)


def test_factory_rabc_auto_detects_from_repo_id():
    """Test that RABC weighter constructs HF path from repo_id."""
    config = SampleWeightingConfig(
        type="rabc",
        progress_path=None,  # Not provided, should auto-detect
    )
    policy = Mock()
    policy.config = Mock()
    policy.config.chunk_size = 50
    device = torch.device("cpu")

    # This will construct the path but fail when trying to load (file doesn't exist)
    # We just verify it doesn't raise the "progress_path required" error
    with pytest.raises(Exception) as exc_info:
        make_sample_weighter(
            config,
            policy,
            device,
            dataset_repo_id="test-user/test-dataset",
        )
    # Should NOT be the "progress_path required" error - it should try to load the file
    assert (
        "progress_path" not in str(exc_info.value).lower() or "auto-detection" in str(exc_info.value).lower()
    )


# =============================================================================
# Integration Tests with RABCWeights
# =============================================================================


def test_rabc_weights_is_sample_weighter(sample_progress_parquet):
    """Test that RABCWeights inherits from SampleWeighter."""
    from lerobot.rewards.sarm.rabc import RABCWeights

    weighter = RABCWeights(
        progress_path=sample_progress_parquet,
        chunk_size=5,
        head_mode="sparse",
    )
    assert isinstance(weighter, SampleWeighter)


def test_rabc_compute_batch_weights(sample_progress_parquet):
    """Test RABCWeights.compute_batch_weights returns correct structure."""
    from lerobot.rewards.sarm.rabc import RABCWeights

    weighter = RABCWeights(
        progress_path=sample_progress_parquet,
        chunk_size=5,
        head_mode="sparse",
        device=torch.device("cpu"),
    )

    batch = {"index": torch.tensor([0, 1, 2, 3])}
    weights, stats = weighter.compute_batch_weights(batch)

    assert isinstance(weights, torch.Tensor)
    assert weights.shape == (4,)
    assert isinstance(stats, dict)
    assert "mean_weight" in stats


def test_rabc_get_stats(sample_progress_parquet):
    """Test RABCWeights.get_stats returns expected structure."""
    from lerobot.rewards.sarm.rabc import RABCWeights

    weighter = RABCWeights(
        progress_path=sample_progress_parquet,
        chunk_size=5,
        head_mode="sparse",
    )

    stats = weighter.get_stats()

    assert stats["type"] == "rabc"
    assert "num_frames" in stats
    assert "chunk_size" in stats
    assert stats["chunk_size"] == 5
    assert "head_mode" in stats
    assert stats["head_mode"] == "sparse"
    assert "delta_mean" in stats
    assert "delta_std" in stats


def test_factory_creates_rabc_weighter(sample_progress_parquet):
    """Test factory creates RABCWeights with valid config."""
    from lerobot.rewards.sarm.rabc import RABCWeights

    config = SampleWeightingConfig(
        type="rabc",
        progress_path=str(sample_progress_parquet),
        head_mode="sparse",
        kappa=0.01,
    )
    policy = Mock()
    policy.config = Mock()
    policy.config.chunk_size = 5
    device = torch.device("cpu")

    weighter = make_sample_weighter(config, policy, device)

    assert isinstance(weighter, RABCWeights)
    assert isinstance(weighter, SampleWeighter)


def test_rabc_weights_normalization(sample_progress_parquet):
    """Test that RABCWeights normalizes weights to sum to batch_size."""
    from lerobot.rewards.sarm.rabc import RABCWeights

    weighter = RABCWeights(
        progress_path=sample_progress_parquet,
        chunk_size=5,
        head_mode="sparse",
        device=torch.device("cpu"),
    )

    batch = {"index": torch.tensor([0, 1, 2, 3])}
    weights, _ = weighter.compute_batch_weights(batch)

    # Weights should be normalized to sum approximately to batch_size
    batch_size = 4
    assert abs(weights.sum().item() - batch_size) < 0.1
