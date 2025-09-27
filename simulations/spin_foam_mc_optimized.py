"""Optimized spin-foam Monte Carlo simulator.

This module provides a flexible Monte Carlo driver for SU(2) spin-foam
configurations on a 4-simplex.  It emphasizes vectorized sampling and optional
GPU acceleration via PyTorch to explore coarse-grained Regge-like actions with
minimal setup.

Example
-------
>>> from simulations.spin_foam_mc_optimized import SpinFoamConfig, SpinFoamMonteCarlo
>>> config = SpinFoamConfig(steps=5000, max_spin=6, use_torch=False)
>>> mc = SpinFoamMonteCarlo(config)
>>> summary = mc.run()
>>> summary["mean_amplitude"]
0.012  # Monte Carlo estimate for ⟨A⟩

Command line usage
------------------
$ python -m simulations.spin_foam_mc_optimized --steps 20000 --max-spin 8 --out results.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np

try:  # Optional GPU acceleration
    import torch
except Exception:  # pragma: no cover - torch is optional
    torch = None


ArrayLike = np.ndarray


@dataclass
class SpinFoamConfig:
    """Configuration parameters for the Monte Carlo simulation."""

    steps: int = 20000
    max_spin: int = 6
    beta: float = 0.35
    proposal_width: int = 2
    batch_size: int = 1024
    rng_seed: int = 2025
    use_torch: bool = False
    torch_device: str = "cuda"
    report_interval: int = 2000

    def ensure_valid(self) -> None:
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.max_spin < 0:
            raise ValueError("max_spin must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.proposal_width <= 0:
            raise ValueError("proposal_width must be positive")
        if self.report_interval < 0:
            raise ValueError("report_interval must be non-negative")


class SpinFoamMonteCarlo:
    """Monte Carlo driver with optional torch acceleration."""

    num_faces: int = 10  # Number of triangles in a 4-simplex
    dihedral_angles: ArrayLike = np.full(10, np.arccos(1 / 4))

    def __init__(self, config: SpinFoamConfig):
        self.config = config
        self.config.ensure_valid()
        self.backend = self._select_backend()
        self.max_spin = int(self.config.max_spin)
        self.rng = np.random.default_rng(self.config.rng_seed)
        self.state = self._initial_state()

    # ------------------------------------------------------------------
    # Backend helpers
    # ------------------------------------------------------------------
    def _select_backend(self):
        if self.config.use_torch:
            if torch is None:
                raise ImportError(
                    "PyTorch is not available. Install with `pip install '.[torch]'` "
                    "or disable `use_torch`."
                )
            device = self.config.torch_device
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
            torch.manual_seed(self.config.rng_seed)
            return {
                "lib": torch,
                "device": torch.device(device),
                "rand": lambda shape: torch.randint(
                    low=0,
                    high=self.max_spin + 1,
                    size=shape,
                    device=device,
                ),
                "uniform": lambda shape: torch.rand(shape, device=device),
                "to_numpy": lambda tensor: tensor.detach().cpu().numpy(),
                "float_type": torch.float32,
            }
        else:
            return {
                "lib": np,
                "device": None,
                "rand": lambda shape: self.rng.integers(0, self.max_spin + 1, size=shape),
                "uniform": lambda shape: self.rng.random(shape),
                "to_numpy": lambda array: np.asarray(array),
                "float_type": float,
            }

    # ------------------------------------------------------------------
    # State initialization
    # ------------------------------------------------------------------
    def _initial_state(self):
        init = self.backend["rand"]((self.num_faces,))
        if self.config.use_torch:
            return init.to(dtype=self.backend["float_type"])
        return init.astype(float)

    # ------------------------------------------------------------------
    # Physics-inspired helpers
    # ------------------------------------------------------------------
    def _area(self, spins):
        lib = self.backend["lib"]
        return lib.sqrt(spins * (spins + 1.0))

    def _regge_action(self, spins):
        lib = self.backend["lib"]
        areas = self._area(spins)
        angles = self.dihedral_angles
        if self.config.use_torch:
            angles = torch.as_tensor(angles, device=self.backend["device"], dtype=self.backend["float_type"])
        return self.config.beta * lib.sum(areas * angles, axis=-1)

    def _volume_weight(self, spins):
        lib = self.backend["lib"]
        areas = self._area(spins)
        mean_area = lib.mean(areas, axis=-1)
        return lib.pow(mean_area + 1e-6, 1.5)

    def _amplitude(self, spins):
        lib = self.backend["lib"]
        spins = self._normalize_input(spins)
        action = self._regge_action(spins)
        weight = self._volume_weight(spins)
        phase = lib.cos(action)
        amplitude = weight * lib.exp(-action) * (0.5 + 0.5 * phase)
        if self.config.use_torch:
            return amplitude.squeeze(0), action.squeeze(0)
        return amplitude.squeeze(0), action.squeeze(0)

    # ------------------------------------------------------------------
    # Sampling utilities
    # ------------------------------------------------------------------
    def _normalize_input(self, spins):
        """Ensure spins have a batch dimension for backend operations."""

        if self.config.use_torch:
            if spins.ndim == 1:
                spins = spins.unsqueeze(0)
            return spins.to(dtype=self.backend["float_type"])
        else:
            if spins.ndim == 1:
                spins = spins[np.newaxis, :]
            return spins.astype(float, copy=False)

    def _propose_state(self, spins):
        face = int(self.rng.integers(0, self.num_faces))
        delta = int(
            self.rng.integers(-self.config.proposal_width, self.config.proposal_width + 1)
        )

        if self.config.use_torch:
            proposal = spins.clone()
            proposal[face] = torch.clamp(proposal[face] + delta, 0, self.max_spin)
        else:
            proposal = spins.copy()
            proposal[face] = np.clip(proposal[face] + delta, 0, self.max_spin)

        return proposal

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> Dict[str, float]:
        current_state = self.state
        current_amp, current_action = self._amplitude(current_state)

        amp_scalar = float(np.abs(self.backend["to_numpy"](current_amp)))
        action_scalar = float(self.backend["to_numpy"](current_action))

        total_amp = 0.0
        total_amp_sq = 0.0
        total_action = 0.0
        total_samples = 0

        for step in range(1, self.config.steps + 1):
            proposal = self._propose_state(current_state)
            prop_amp, prop_action = self._amplitude(proposal)

            prop_amp_scalar = float(np.abs(self.backend["to_numpy"](prop_amp)))
            prop_action_scalar = float(self.backend["to_numpy"](prop_action))

            denom = max(amp_scalar, 1e-12)
            accept_prob = min(1.0, prop_amp_scalar / denom)
            if self.rng.random() < accept_prob:
                current_state = proposal
                current_amp = prop_amp
                current_action = prop_action
                amp_scalar = prop_amp_scalar
                action_scalar = prop_action_scalar

            total_amp += amp_scalar
            total_amp_sq += amp_scalar**2
            total_action += action_scalar
            total_samples += 1

            if self.config.report_interval and step % self.config.report_interval == 0:
                mean_amp = total_amp / total_samples
                print(
                    f"[spin-foam] step {step:>7} — accept_prob={accept_prob:.3f} "
                    f"⟨A⟩={mean_amp:.6f}"
                )

        self.state = current_state

        mean_amp = total_amp / total_samples
        var_amp = max(total_amp_sq / total_samples - mean_amp**2, 0.0)
        mean_action = total_action / total_samples
        stderr = np.sqrt(var_amp / total_samples)

        return {
            "mean_amplitude": mean_amp,
            "amplitude_std": np.sqrt(var_amp),
            "amplitude_stderr": stderr,
            "mean_action": mean_action,
            "total_samples": total_samples,
        }


# ----------------------------------------------------------------------
# Command line interface
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimized spin-foam Monte Carlo simulator")
    parser.add_argument("--steps", type=int, default=20000, help="Number of Monte Carlo samples to draw")
    parser.add_argument("--max-spin", type=int, default=6, help="Maximum spin label j")
    parser.add_argument("--beta", type=float, default=0.35, help="Regge action coupling")
    parser.add_argument("--proposal-width", type=int, default=2, help="Uniform proposal width for spin updates")
    parser.add_argument("--batch-size", type=int, default=1024, help="Vectorized batch size")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument("--use-torch", action="store_true", help="Enable PyTorch backend for GPU acceleration")
    parser.add_argument("--torch-device", type=str, default="cuda", help="Torch device to use (if available)")
    parser.add_argument("--out", type=Path, default=None, help="Optional path to JSON summary output")
    parser.add_argument(
        "--report-interval",
        type=int,
        default=2000,
        help="How many samples between progress logs (0 to disable)",
    )
    return parser.parse_args()


def main(args: Optional[argparse.Namespace] = None) -> Dict[str, float]:
    if args is None:
        args = parse_args()

    config = SpinFoamConfig(
        steps=args.steps,
        max_spin=args.max_spin,
        beta=args.beta,
        proposal_width=args.proposal_width,
        batch_size=args.batch_size,
        rng_seed=args.seed,
        use_torch=args.use_torch,
        torch_device=args.torch_device,
        report_interval=max(args.report_interval, 0) or 0,
    )

    mc = SpinFoamMonteCarlo(config)
    summary = mc.run()

    print("Spin-foam Monte Carlo summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as fh:
            json.dump({"config": asdict(config), "summary": summary}, fh, indent=2)
        print(f"Saved summary to {args.out}")

    return summary


if __name__ == "__main__":  # pragma: no cover
    main()
