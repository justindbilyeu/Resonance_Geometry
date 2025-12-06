"""
Resonance Geometry: Science Suite
---------------------------------
Automated parameter sweeps for v2.1 Canon.
Targets: Robustness of Memory & Functional Gain.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from .resonance_universe import ResonanceUniverse

def measure_performance(alpha, beta, N=10, trials=5):
    memory_scores = []
    functional_gains = []

    for _ in range(trials):
        uni = ResonanceUniverse(N=N, seed=None)
        uni.run_lifecycle(T_max=600, dt=0.1)

        # Metric A: Memory Strength (Final Phase C Phi)
        memory_scores.append(np.mean(uni.history['phi'][-50:]))

        # Metric B: Functional Gain (Exp D)
        resp_trained = uni.probe_response(drive_freq=0.0)

        uni_rand = ResonanceUniverse(N=N, seed=None)
        target_mean_K = np.mean(uni.K)
        uni_rand.K = np.random.uniform(0, 2*target_mean_K, (N,N))
        uni_rand.K = (uni_rand.K + uni_rand.K.T)/2
        np.fill_diagonal(uni_rand.K, 0)
        resp_rand = uni_rand.probe_response(drive_freq=0.0)

        functional_gains.append(resp_trained / (resp_rand + 1e-6))

    return np.mean(memory_scores), np.mean(functional_gains)

def run_alpha_beta_sweep():
    print("Starting Alpha-Beta Sweep...")
    alphas = np.linspace(0.01, 0.2, 5)
    betas = np.linspace(0.1, 2.0, 5)

    heatmap_mem = np.zeros((len(alphas), len(betas)))
    heatmap_gain = np.zeros((len(alphas), len(betas)))

    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            print(f"  Testing alpha={a:.2f}, beta={b:.2f}...")
            mem, gain = measure_performance(alpha=a, beta=b)
            heatmap_mem[i, j] = mem
            heatmap_gain[i, j] = gain

    # Plotting
    output_dir = "experiments/outputs/toy_model"
    os.makedirs(output_dir, exist_ok=True)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    im1 = axs[0].imshow(heatmap_mem, origin='lower', extent=[0.1, 2.0, 0.01, 0.2], aspect='auto', cmap='viridis')
    axs[0].set_title("Memory Robustness"); axs[0].set_xlabel("Beta"); axs[0].set_ylabel("Alpha")
    plt.colorbar(im1, ax=axs[0])

    im2 = axs[1].imshow(heatmap_gain, origin='lower', extent=[0.1, 2.0, 0.01, 0.2], aspect='auto', cmap='plasma')
    axs[1].set_title("Functional Gain"); axs[1].set_xlabel("Beta")
    plt.colorbar(im2, ax=axs[1])

    plt.savefig(f"{output_dir}/sweep_results.png")
    print(f"Saved sweep to {output_dir}/sweep_results.png")

if __name__ == "__main__":
    run_alpha_beta_sweep()
