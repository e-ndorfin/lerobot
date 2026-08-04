# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""
Sample weighting abstraction for training.

This module provides an abstract base class for sample weighting strategies (e.g., RA-BC)
that can be used during training without polluting the training script with
policy-specific code.

Example usage:
    # In training config
    sample_weighting:
        type: rabc
        progress_path: hf://datasets/my-dataset/sarm_progress.parquet
        head_mode: sparse
        kappa: 0.01

    # In training script
    sample_weighter = make_sample_weighter(
        cfg.sample_weighting,
        policy,
        device,
        dataset_root=cfg.dataset.root,
        dataset_repo_id=cfg.dataset.repo_id,
    )
    ...
    weights, stats = sample_weighter.compute_batch_weights(batch)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch

CONTROL_MODE_KEY = "observation.control_mode"

if TYPE_CHECKING:
    from lerobot.policies.pretrained import PreTrainedPolicy


class SampleWeighter(ABC):
    """
    Implementations compute per-sample weights that can be used to weight
    the loss during training. This enables techniques like:
    - RA-BC (Reward-Aligned Behavior Cloning)
    - Importance sampling
    - Curriculum learning
    - Quality-based filtering
    """

    @abstractmethod
    def compute_batch_weights(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """
        Compute per-sample weights for a training batch.

        Args:
            batch: Training batch dictionary containing at minimum an "index" key
                   with global frame indices.
        """

    @abstractmethod
    def get_stats(self) -> dict:
        """
        Get global statistics about the weighting strategy.
        """


@dataclass
class SampleWeightingConfig:
    """
    Configuration for sample weighting during training.

    This is a generic config that supports multiple weighting strategies.
    The `type` field determines which implementation to use, and `extra_params`
    contains additional type-specific parameters.

    Attributes:
        type: Weighting strategy type ("rabc", "control_mode", "uniform", etc.)
        progress_path: Path to precomputed progress values (for RABC)
        head_mode: Which model head to use for progress ("sparse" or "dense")
        kappa: Hard threshold for high-quality samples (RABC-specific)
        epsilon: Small constant for numerical stability
        mode_weights: Mapping from integer control-mode labels to BC loss weights.
        control_mode_key: Raw dataset feature containing the control-mode label.
        default_weight: Weight for labels omitted from ``mode_weights``. ``None`` rejects unknown labels.
        extra_params: Additional type-specific parameters passed to the weighter
    """

    type: str = "rabc"
    progress_path: str | None = None
    head_mode: str = "sparse"
    kappa: float = 0.01
    epsilon: float = 1e-6
    mode_weights: dict[int, float] = field(default_factory=dict)
    control_mode_key: str = CONTROL_MODE_KEY
    default_weight: float | None = None
    # Additional type-specific params can be added here or passed via extra_params
    extra_params: dict = field(default_factory=dict)


def make_sample_weighter(
    config: SampleWeightingConfig | None,
    policy: PreTrainedPolicy,
    device: torch.device,
    dataset_root: str | None = None,
    dataset_repo_id: str | None = None,
) -> SampleWeighter | None:
    """
    Factory function to create a SampleWeighter from config.

    This keeps policy-specific initialization logic out of the training script.

    Args:
        config: Sample weighting configuration, or None to disable weighting.
        policy: The policy being trained (used to extract chunk_size, etc.)
        device: Device to place weight tensors on.
        dataset_root: Local path to dataset root (for auto-detecting progress_path).
        dataset_repo_id: HuggingFace repo ID (for auto-detecting progress_path).
    """
    if config is None:
        return None

    if config.type == "rabc":
        return _make_rabc_weighter(config, policy, device, dataset_root, dataset_repo_id)

    if config.type == "uniform":
        # No-op weighter that returns uniform weights
        return UniformWeighter(device=device)

    if config.type == "control_mode":
        return ControlModeWeighter(
            mode_weights=config.mode_weights,
            device=device,
            control_mode_key=config.control_mode_key,
            default_weight=config.default_weight,
        )

    raise ValueError(
        f"Unknown sample weighting type: '{config.type}'. Supported types: 'rabc', 'control_mode', 'uniform'"
    )


def _make_rabc_weighter(
    config: SampleWeightingConfig,
    policy: PreTrainedPolicy,
    device: torch.device,
    dataset_root: str | None = None,
    dataset_repo_id: str | None = None,
) -> SampleWeighter:
    """Create RABC weighter with policy-specific initialization.

    Args:
        config: Sample weighting configuration.
        policy: The policy being trained (used to extract chunk_size).
        device: Device to place weight tensors on.
        dataset_root: Local path to dataset root (for auto-detecting progress_path).
        dataset_repo_id: HuggingFace repo ID (for auto-detecting progress_path).
    """
    # Import here to avoid circular imports and keep RABC code in SARM module
    from lerobot.rewards.sarm.rabc import RABCWeights

    # Extract chunk_size from policy config
    chunk_size = getattr(policy.config, "chunk_size", None)
    if chunk_size is None:
        raise ValueError(
            "RABC sample weighting requires a policy with 'chunk_size' in its config. "
            "This is typically set for action-chunking policies like ACT, Diffusion, PI0, etc."
        )

    # Determine progress_path: use explicit config or auto-detect from dataset
    progress_path = config.progress_path
    if progress_path is None:
        if dataset_root:
            progress_path = str(Path(dataset_root) / "sarm_progress.parquet")
        elif dataset_repo_id:
            progress_path = f"hf://datasets/{dataset_repo_id}/sarm_progress.parquet"
        else:
            raise ValueError(
                "RABC sample weighting requires 'progress_path' to be set, "
                "or dataset_root/dataset_repo_id for auto-detection. "
                "Generate progress values using: "
                "python -m lerobot.rewards.sarm.compute_rabc_weights --help"
            )

    return RABCWeights(
        progress_path=progress_path,
        chunk_size=chunk_size,
        head_mode=config.head_mode,
        kappa=config.kappa,
        epsilon=config.epsilon,
        device=device,
        **config.extra_params,
    )


class UniformWeighter(SampleWeighter):
    """
    No-op sample weighter that returns uniform weights.

    Useful as a baseline or when you want to disable weighting without
    changing the training code structure.

    Note:
        Batch size is determined by looking for tensor values in the batch
        dictionary. The method checks common keys like "action", "index",
        and "observation.state" first, then falls back to scanning all values.
    """

    def __init__(self, device: torch.device):
        self.device = device

    def compute_batch_weights(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """Return uniform weights (all ones)."""
        batch_size = self._determine_batch_size(batch)

        weights = torch.ones(batch_size, device=self.device)
        stats = {"mean_weight": 1.0, "type": "uniform"}
        return weights, stats

    def _determine_batch_size(self, batch: dict) -> int:
        """
        Determine batch size from the batch dictionary.

        Checks common keys first, then scans all values for tensors.

        Args:
            batch: Training batch dictionary.
        """
        if not batch:
            raise ValueError("Cannot determine batch size from empty batch")

        # Check common keys first
        for key in ["action", "index", "observation.state"]:
            if key in batch and isinstance(batch[key], torch.Tensor):
                return batch[key].shape[0]

        # Scan all values for any tensor
        for value in batch.values():
            if isinstance(value, torch.Tensor) and value.ndim >= 1:
                return value.shape[0]

        # Last resort: return 1 (this handles non-tensor batches)
        return 1

    def get_stats(self) -> dict:
        """Return empty stats for uniform weighting."""
        return {"type": "uniform"}


class ControlModeWeighter(SampleWeighter):
    """Assign a BC loss weight from a raw per-frame control-mode label.

    The returned weights are normalized to sum to the batch size when at least one
    sample has non-zero weight. This keeps gradient scale stable while preserving
    the configured weight ratios. A batch containing only zero-weight modes remains
    all-zero and causes the training loop to skip its optimizer update.
    """

    def __init__(
        self,
        mode_weights: dict[int, float],
        device: torch.device,
        control_mode_key: str = CONTROL_MODE_KEY,
        default_weight: float | None = None,
    ) -> None:
        if not mode_weights:
            raise ValueError("control_mode sample weighting requires a non-empty mode_weights mapping")
        if not control_mode_key:
            raise ValueError("control_mode_key must not be empty")

        self.mode_weights = {
            self._validate_mode(mode): self._validate_weight(weight, f"mode {mode}")
            for mode, weight in mode_weights.items()
        }
        if len(self.mode_weights) != len(mode_weights):
            raise ValueError(f"mode_weights contains duplicate integer labels: {mode_weights}")
        self.default_weight = (
            None if default_weight is None else self._validate_weight(default_weight, "default_weight")
        )
        self.device = device
        self.control_mode_key = control_mode_key
        self._mode_counts: Counter[int] = Counter()
        self._num_batches = 0
        self._num_zero_weight_batches = 0

    @staticmethod
    def _validate_mode(mode: int) -> int:
        if not isinstance(mode, int) or isinstance(mode, bool):
            raise ValueError(f"control-mode labels must be integers, got {mode!r}")
        return mode

    @staticmethod
    def _validate_weight(weight: float, name: str) -> float:
        if not isinstance(weight, (int, float)) or isinstance(weight, bool):
            raise ValueError(f"Weight for {name} must be numeric, got {weight!r}")
        value = float(weight)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Weight for {name} must be finite and non-negative, got {weight!r}")
        return value

    def compute_batch_weights(self, batch: dict) -> tuple[torch.Tensor, dict]:
        if self.control_mode_key not in batch:
            raise ValueError(
                f"control_mode sample weighting requires raw batch feature {self.control_mode_key!r}"
            )

        modes = torch.as_tensor(batch[self.control_mode_key]).detach()
        if modes.ndim == 0:
            modes = modes.unsqueeze(0)
        if modes.ndim == 2 and modes.shape[1] == 1:
            modes = modes[:, 0]
        if modes.ndim != 1:
            raise ValueError(
                f"{self.control_mode_key} must contain one scalar label per sample, "
                f"got shape {tuple(modes.shape)}"
            )
        if not torch.isfinite(modes).all():
            raise ValueError(f"{self.control_mode_key} contains non-finite labels")

        rounded_modes = modes.round()
        if not torch.allclose(modes.to(torch.float64), rounded_modes.to(torch.float64)):
            invalid = modes[modes != rounded_modes].tolist()
            raise ValueError(f"{self.control_mode_key} contains non-integer labels: {invalid}")
        mode_ids = rounded_modes.to(dtype=torch.int64, device="cpu")

        observed_modes = sorted(set(mode_ids.tolist()))
        unknown_modes = [mode for mode in observed_modes if mode not in self.mode_weights]
        if unknown_modes and self.default_weight is None:
            raise ValueError(
                f"No BC weight configured for control modes {unknown_modes}; configured modes are "
                f"{sorted(self.mode_weights)}. Set sample_weighting.default_weight to allow unknown modes."
            )

        fallback = 0.0 if self.default_weight is None else self.default_weight
        device_mode_ids = mode_ids.to(self.device)
        weights = torch.full((len(mode_ids),), fallback, dtype=torch.float32, device=self.device)
        for mode, weight in self.mode_weights.items():
            weights[device_mode_ids == mode] = weight

        raw_weight_sum = weights.sum()
        zero_weight_batch = bool(raw_weight_sum.item() == 0)
        if not zero_weight_batch:
            weights = weights * (len(weights) / raw_weight_sum)

        counts = Counter(mode_ids.tolist())
        self._mode_counts.update(counts)
        self._num_batches += 1
        self._num_zero_weight_batches += int(zero_weight_batch)

        stats = {
            "type": "control_mode",
            "mean_weight": weights.mean().item() if len(weights) else 0.0,
            "min_weight": weights.min().item() if len(weights) else 0.0,
            "max_weight": weights.max().item() if len(weights) else 0.0,
            "zero_weight_batch": int(zero_weight_batch),
        }
        for mode, count in sorted(counts.items()):
            stats[f"mode_{mode}_count"] = count
        return weights, stats

    def get_stats(self) -> dict:
        stats = {
            "type": "control_mode",
            "control_mode_key": self.control_mode_key,
            "num_batches": self._num_batches,
            "zero_weight_batches": self._num_zero_weight_batches,
        }
        for mode, weight in sorted(self.mode_weights.items()):
            stats[f"mode_{mode}_weight"] = weight
            stats[f"mode_{mode}_samples"] = self._mode_counts[mode]
        if self.default_weight is not None:
            stats["default_weight"] = self.default_weight
        return stats
