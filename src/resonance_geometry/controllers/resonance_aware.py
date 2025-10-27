#!/usr/bin/env python3
"""
Resonance-Aware Inference Controller (RAIC) v3

A runtime controller that monitors λ_max (leading eigenvalue) of the covariance
matrix during inference and applies adaptive interventions when instability
is detected.

Key features:
- Rolling window estimation of λ_max statistics
- σ-threshold detection (mean + k*std)
- Configurable interventions: temperature scaling, top-k filtering, prompt injection
- Minimal overhead for production deployment

Attribution: Inspired by spectral stability analysis in hallucination theory
"""

import numpy as np
from collections import deque
from typing import Optional, Dict, Any, List


class ResonanceAwareController:
    """
    Runtime controller for detecting and mitigating hallucination risk
    via spectral stability monitoring.

    Usage:
        controller = ResonanceAwareController(window_size=50, threshold_sigma=2.0)

        for step in inference_loop:
            activations = model.get_activations(layer_idx)
            lambda_max = controller.compute_lambda_max(activations)

            if controller.should_intervene():
                temperature = controller.get_adaptive_temperature()
                # Apply intervention...
    """

    def __init__(
        self,
        window_size: int = 50,
        threshold_sigma: float = 2.0,
        intervention_mode: str = "temperature",
        min_temperature: float = 0.3,
        max_temperature: float = 1.0,
        baseline_temperature: float = 0.7,
        layer_indices: Optional[List[int]] = None
    ):
        """
        Initialize RAIC controller.

        Args:
            window_size: Number of recent λ_max values to track
            threshold_sigma: Number of standard deviations for alarm threshold
            intervention_mode: "temperature", "top_k", "prompt", or "hybrid"
            min_temperature: Lower bound for temperature scaling
            max_temperature: Upper bound for temperature scaling
            baseline_temperature: Default temperature when stable
            layer_indices: Which layers to monitor (None = all middle layers)
        """
        self.window_size = window_size
        self.threshold_sigma = threshold_sigma
        self.intervention_mode = intervention_mode
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.baseline_temperature = baseline_temperature
        self.layer_indices = layer_indices

        # Rolling statistics
        self.lambda_history = deque(maxlen=window_size)
        self.instability_count = 0
        self.total_steps = 0

        # State tracking
        self.current_lambda_max = 0.0
        self.current_temperature = baseline_temperature
        self.alarm_threshold = 0.0
        self.is_alarmed = False

    def compute_lambda_max(self, covariance_matrix: np.ndarray) -> float:
        """
        Compute leading eigenvalue of covariance matrix.

        Args:
            covariance_matrix: (d, d) covariance or correlation matrix

        Returns:
            λ_max: Largest eigenvalue
        """
        try:
            # Use eigvalsh for Hermitian matrices (faster and more stable)
            evals = np.linalg.eigvalsh(covariance_matrix)
            lambda_max = float(np.max(evals))
        except np.linalg.LinAlgError:
            # Fallback to regularized version
            eps = 1e-6
            reg_cov = covariance_matrix + eps * np.eye(covariance_matrix.shape[0])
            evals = np.linalg.eigvalsh(reg_cov)
            lambda_max = float(np.max(evals))

        return lambda_max

    def update(self, lambda_max: float) -> Dict[str, Any]:
        """
        Update controller state with new λ_max observation.

        Args:
            lambda_max: Current leading eigenvalue

        Returns:
            status: Dict with alarm status, threshold, temperature, etc.
        """
        self.current_lambda_max = lambda_max
        self.lambda_history.append(lambda_max)
        self.total_steps += 1

        # Compute rolling statistics (need at least 3 samples for meaningful std)
        if len(self.lambda_history) >= 3:
            hist_array = np.array(self.lambda_history)
            mean_lambda = float(np.mean(hist_array))
            std_lambda = float(np.std(hist_array, ddof=1))
            self.alarm_threshold = mean_lambda + self.threshold_sigma * std_lambda

            # Check if current value exceeds threshold
            self.is_alarmed = (lambda_max > self.alarm_threshold)

            if self.is_alarmed:
                self.instability_count += 1
        else:
            # Not enough data yet
            self.alarm_threshold = float('inf')
            self.is_alarmed = False

        # Compute adaptive temperature if alarmed
        if self.is_alarmed:
            self.current_temperature = self._compute_adaptive_temperature()
        else:
            self.current_temperature = self.baseline_temperature

        return self.get_status()

    def _compute_adaptive_temperature(self) -> float:
        """
        Compute temperature intervention based on alarm severity.

        Strategy: Map excess λ_max above threshold to temperature reduction.
        Higher λ_max → lower temperature (more conservative sampling)
        """
        if len(self.lambda_history) < 3:
            return self.baseline_temperature

        hist_array = np.array(self.lambda_history)
        std_lambda = float(np.std(hist_array, ddof=1))

        if std_lambda < 1e-9:
            return self.baseline_temperature

        # Compute normalized excess (how many σ above threshold)
        excess = (self.current_lambda_max - self.alarm_threshold) / (std_lambda + 1e-9)

        # Map excess to temperature reduction
        # excess = 0 → baseline, excess > 2 → min_temperature
        # Linear interpolation with saturation
        if excess <= 0:
            temperature = self.baseline_temperature
        else:
            # Decrease temperature linearly with excess
            alpha = min(excess / 2.0, 1.0)  # Saturate at 2σ above threshold
            temperature = self.baseline_temperature - alpha * (self.baseline_temperature - self.min_temperature)
            temperature = max(self.min_temperature, min(self.max_temperature, temperature))

        return temperature

    def should_intervene(self) -> bool:
        """Check if intervention is needed based on current alarm state."""
        return self.is_alarmed

    def get_adaptive_temperature(self) -> float:
        """Get current temperature recommendation."""
        return self.current_temperature

    def get_status(self) -> Dict[str, Any]:
        """Get current controller status."""
        return {
            'lambda_max': self.current_lambda_max,
            'alarm_threshold': self.alarm_threshold,
            'is_alarmed': self.is_alarmed,
            'temperature': self.current_temperature,
            'instability_rate': self.instability_count / max(self.total_steps, 1),
            'steps': self.total_steps,
            'history_size': len(self.lambda_history)
        }

    def reset(self):
        """Reset controller state (e.g., at start of new sequence)."""
        self.lambda_history.clear()
        self.instability_count = 0
        self.total_steps = 0
        self.current_lambda_max = 0.0
        self.current_temperature = self.baseline_temperature
        self.alarm_threshold = 0.0
        self.is_alarmed = False


class BatchRAICHarness:
    """
    Batch evaluation harness for RAIC on synthetic or real data.

    Simulates multi-step inference with interventions and tracks:
    - Alarm frequency
    - Temperature adjustments
    - Downstream accuracy (if ground truth available)
    """

    def __init__(self, controller: ResonanceAwareController):
        self.controller = controller
        self.log = []

    def run_sequence(
        self,
        covariance_matrices: List[np.ndarray],
        reset_between: bool = True
    ) -> Dict[str, Any]:
        """
        Run RAIC on a sequence of covariance matrices.

        Args:
            covariance_matrices: List of (d, d) covariance matrices (one per step)
            reset_between: Whether to reset controller between sequences

        Returns:
            summary: Dict with alarm counts, temperatures, etc.
        """
        if reset_between:
            self.controller.reset()

        for step_idx, cov in enumerate(covariance_matrices):
            lambda_max = self.controller.compute_lambda_max(cov)
            status = self.controller.update(lambda_max)

            self.log.append({
                'step': step_idx,
                **status
            })

        # Compute summary statistics
        alarm_count = sum(1 for entry in self.log if entry['is_alarmed'])
        avg_temperature = np.mean([entry['temperature'] for entry in self.log])
        max_lambda = max(entry['lambda_max'] for entry in self.log)

        return {
            'total_steps': len(covariance_matrices),
            'alarm_count': alarm_count,
            'alarm_rate': alarm_count / len(covariance_matrices),
            'avg_temperature': avg_temperature,
            'max_lambda_max': max_lambda,
            'log': self.log
        }

    def get_log(self) -> List[Dict[str, Any]]:
        """Return full step-by-step log."""
        return self.log
