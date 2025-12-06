"""
Resonance Geometry: Toy Universe (v2.1)
---------------------------------------
A rigorous implementation of Geometric Plasticity on a Kuramoto substrate.
Formalism aligned with "Resonance Geometry" (Gemini/Claude Co-Development).

Definitions:
- Base Manifold (M): N-Torus T^N (Phases theta)
- Geometry Manifold (G): Symmetric, Non-negative Matrices (Coupling K)
- Combined State Space (S): M x G
- Energy Functional (L): Stress Energy minimizing conflict between geometry and resonance.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import shortest_path, laplacian
from scipy.linalg import eigh
from sklearn.decomposition import PCA
import os

class ResonanceUniverse:
    def __init__(self, N=10, seed=42):
        if seed is not None:
            np.random.seed(seed)
        self.N = N

        # --- 1. State Space M (The Territory) ---
        # Intrinsic properties: Natural frequencies ~ N(0, 2.0)
        self.omegas = np.random.normal(loc=0.0, scale=2.0, size=N)
        # State: Phases theta
        self.theta = np.random.uniform(0, 2*np.pi, N)

        # --- 2. Geometry Space G (The Map) ---
        # State: Coupling Matrix K
        # Constraint: Symmetric, Non-negative.
        # Initial: Weak random coupling (Erdos-Renyi-like fuzz)
        K_init = np.random.uniform(0, 0.5, (N, N))
        self.K = (K_init + K_init.T) / 2
        np.fill_diagonal(self.K, 0)

        # --- 3. Telemetry ---
        self.history = {
            'phi': [],          # Order parameter (Global Synchrony)
            'stress': [],       # Potential Energy L
            'lambda_2': [],     # Fiedler Value (Algebraic Connectivity)
            'mean_K': []        # Average coupling strength
        }

    def _compute_order_parameter(self, theta):
        """Scalar measure of synchrony Phi in [0, 1]."""
        z = np.mean(np.exp(1j * theta))
        return np.abs(z)

    def _compute_energy(self, beta):
        """
        Calculates the Free Energy Functional L(theta, K).
        L = Potential(theta, K) + Regularizer(K)
        """
        delta_theta = self.theta[None, :] - self.theta[:, None]
        # Potential: V = - sum(K * cos(delta_theta))
        alignment = np.sum(self.K * np.cos(delta_theta))
        # Regularizer: R = (beta/2) * ||K||^2
        metabolic = (beta / 2.0) * np.sum(self.K**2)
        return metabolic - alignment

    def _compute_spectrum(self):
        """
        Computes the Laplacian Spectrum of the current geometry.
        Returns lambda_2 (Fiedler value).
        lambda_2 > 0 implies a connected component.
        """
        # Combinatorial Laplacian L = D - K
        L = np.diag(np.sum(self.K, axis=1)) - self.K
        # Eigenvalues, sorted smallest to largest
        evals = eigh(L, eigvals_only=True)
        # evals[0] is always ~0. evals[1] is Fiedler.
        return evals[1] if len(evals) > 1 else 0.0

    def step(self, dt, plasticity_on=False, alpha=0.1, beta=0.5):
        """
        Evolve the Combined State S = M x G by one step dt.
        Dynamics are Gradient Descent on L (approx).
        """
        # --- 1. Fast Dynamics on M (Phase Flow) ---
        delta_theta = self.theta[None, :] - self.theta[:, None]
        interaction = np.sin(delta_theta)
        d_theta = self.omegas + (1.0/self.N) * np.sum(self.K * interaction, axis=1)

        # --- 2. Slow Dynamics on G (Geometric Flow) ---
        if plasticity_on:
            # Gradient Descent: dK = alpha * (cos(delta_theta) - beta * K)
            correlation = np.cos(delta_theta)
            d_K = alpha * (correlation - beta * self.K)

            self.K += d_K * dt

            # --- Enforce Constraints on G ---
            self.K = (self.K + self.K.T) / 2 # Symmetry
            self.K[self.K < 0] = 0           # Non-negativity
            np.fill_diagonal(self.K, 0)      # No self-loops

        # Update Phases
        self.theta += d_theta * dt

        # --- 3. Log Telemetry ---
        self.history['phi'].append(self._compute_order_parameter(self.theta))
        self.history['stress'].append(self._compute_energy(beta))
        self.history['mean_K'].append(np.mean(self.K))
        self.history['lambda_2'].append(self._compute_spectrum())

    def run_lifecycle(self, T_max=800, dt=0.05):
        """
        Runs Experiments A, B, and C in sequence.
        A: Baseline (0-200), B: Learning (200-600), C: Memory (600-800)
        """
        steps = int(T_max/dt)
        t_phase_A = 200
        t_phase_B = 600

        print(f"Running Lifecycle (T={T_max})...")
        for s in range(steps):
            t = s * dt
            plasticity = (t > t_phase_A and t <= t_phase_B)

            # Perturbation Event (Start of Phase C)
            if abs(t - t_phase_B) < dt/2:
                self.theta = np.random.uniform(0, 2*np.pi, self.N)

            self.step(dt, plasticity_on=plasticity)

    def probe_response(self, drive_freq=0.0, strength=0.5, duration=200, dt=0.05):
        """
        Experiment D: Signal Injection.
        Freezes geometry and probes response to external drive.
        Returns: Average Order Parameter during drive.
        """
        saved_theta = self.theta.copy()
        self.theta = np.random.uniform(0, 2*np.pi, self.N) # Reset phases

        phi_response = []
        steps = int(duration / dt)

        for s in range(steps):
            t = s * dt
            drive_force = strength * np.sin(drive_freq * t - self.theta)

            delta_theta = self.theta[None, :] - self.theta[:, None]
            interaction = np.sin(delta_theta)
            d_theta = self.omegas + (1.0/self.N) * np.sum(self.K * interaction, axis=1) + drive_force

            self.theta += d_theta * dt
            phi_response.append(self._compute_order_parameter(self.theta))

        avg_response = np.mean(phi_response[int(steps/2):])
        self.theta = saved_theta
        return avg_response

if __name__ == "__main__":
    # Ensure output dir exists
    output_dir = "experiments/outputs/toy_model"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Initialize & Run Lifecycle
    uni = ResonanceUniverse(N=10)
    uni.run_lifecycle()

    # 2. Experiment D (Probe)
    resp_trained = uni.probe_response(drive_freq=0.0)

    uni_rand = ResonanceUniverse(N=10)
    target_mean_K = np.mean(uni.K)
    uni_rand.K = np.random.uniform(0, 2*target_mean_K, (10,10))
    uni_rand.K = (uni_rand.K + uni_rand.K.T)/2
    np.fill_diagonal(uni_rand.K, 0)
    resp_rand = uni_rand.probe_response(drive_freq=0.0)

    print(f"Exp D Results: Trained={resp_trained:.4f}, Random={resp_rand:.4f}")

    # 3. Visualization
    time = np.linspace(0, 800, len(uni.history['phi']))
    fig, axs = plt.subplots(4, 1, figsize=(10, 14), sharex=True)

    # Plots
    axs[0].plot(time, uni.history['phi'], color='purple', lw=2); axs[0].set_ylabel("Synchrony Phi")
    axs[1].plot(time, uni.history['stress'], color='red', lw=2); axs[1].set_ylabel("Free Energy L")
    axs[2].plot(time, uni.history['lambda_2'], color='blue', lw=2); axs[2].set_ylabel("Fiedler Val")
    axs[3].plot(time, uni.history['mean_K'], color='green', lw=2); axs[3].set_ylabel("Mean Coupling")

    plt.savefig(f"{output_dir}/lifecycle_plot.png")
    print(f"Saved plot to {output_dir}/lifecycle_plot.png")
