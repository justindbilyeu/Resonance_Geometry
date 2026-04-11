#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import odeint
from pathlib import Path
import json
from datetime import datetime
import sys

GAMMA_BASE = 1.0
I_EFF = 0.6
ALPHA = 0.25
BETA = 0.01
SKEW = 0.05
K = 1.0

ETA_VALUES = np.array([0, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0])
LAMBDA_VALUES = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
ETA_HYSTERESIS = np.linspace(0.1, 5.0, 50)
LAMBDA_FIXED = 1.0
GAMMA_FIXED = 1.0

T_MAX = 150.0
DT = 0.05
T_SPAN = np.arange(0, T_MAX, DT)
OMEGA_0 = np.array([0.12, 0.08, 0.05])

np.random.seed(42)

def master_equation_sage(omega, t, eta, lam, gamma, I_eff=0.6, alpha=0.25, beta=0.01, skew=0.05, k=1.0):
    drive = eta * I_eff * omega
    grounding = -lam * k * omega
    damping = -gamma * omega
    r2 = np.dot(omega, omega)
    r4 = r2 * r2
    nonlinear = alpha * r2 * omega - beta * r4 * omega
    skew_coupling = skew * np.array([omega[1], -omega[0], 0])
    return drive + grounding + damping + nonlinear + skew_coupling

def compute_stability_sage(omega, eta, lam, gamma, I_eff=0.6, alpha=0.25, beta=0.01, skew=0.05, k=1.0):
    eps = 1e-6
    J = np.zeros((3, 3))
    for i in range(3):
        omega_plus = omega.copy()
        omega_plus[i] += eps
        omega_minus = omega.copy()
        omega_minus[i] -= eps
        f_plus = master_equation_sage(omega_plus, 0, eta, lam, gamma, I_eff, alpha, beta, skew, k)
        f_minus = master_equation_sage(omega_minus, 0, eta, lam, gamma, I_eff, alpha, beta, skew, k)
        J[:, i] = (f_plus - f_minus) / (2 * eps)
    eigenvalues = np.linalg.eigvals(J)
    return np.max(eigenvalues.real)

def classify_regime(lambda_max):
    if lambda_max < -0.05:
        return "grounded"
    elif lambda_max <= 0.05:
        return "creative"
    else:
        return "hallucinatory"

def task_1_phase_diagram():
    print("\nPHASE DIAGRAM")
    Path("rg/results/sage_corrected").mkdir(parents=True, exist_ok=True)
    
    results_grid = np.zeros((len(ETA_VALUES), len(LAMBDA_VALUES)))
    regime_grid = np.empty((len(ETA_VALUES), len(LAMBDA_VALUES)), dtype=object)
    
    for i, eta in enumerate(ETA_VALUES):
        for j, lam in enumerate(LAMBDA_VALUES):
            omega_0 = OMEGA_0 + 0.05 * np.random.randn(3)
            trajectory = odeint(master_equation_sage, omega_0, T_SPAN, args=(eta, lam, GAMMA_BASE, I_EFF, ALPHA, BETA, SKEW, K))
            omega_final = np.mean(trajectory[-int(0.2*len(trajectory)):], axis=0)
            lambda_max = compute_stability_sage(omega_final, eta, lam, GAMMA_BASE, I_EFF, ALPHA, BETA, SKEW, K)
            regime = classify_regime(lambda_max)
            results_grid[i, j] = lambda_max
            regime_grid[i, j] = regime
            if (i * len(LAMBDA_VALUES) + j + 1) % 9 == 0:
                print(f"  Progress: eta={eta:.1f} lambda={lam:.1f} -> {lambda_max:.3f} ({regime})")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(results_grid, xticklabels=np.round(LAMBDA_VALUES, 2), yticklabels=np.round(ETA_VALUES, 2), cmap="RdBu_r", center=0, ax=ax)
    ax.contour(results_grid, levels=[0], colors="black", linewidths=3)
    ax.set_xlabel("lambda (Grounding)")
    ax.set_ylabel("eta (Meta-Awareness)")
    ax.set_title("Phase Diagram: Sage-Corrected")
    plt.tight_layout()
    plt.savefig("rg/results/sage_corrected/phase_diagram.png", dpi=300)
    plt.close()
    
    n_halluc = np.sum(regime_grid == "hallucinatory")
    print(f"Hallucinatory: {n_halluc}/{len(ETA_VALUES)*len(LAMBDA_VALUES)}")
    return results_grid, n_halluc > 0

def task_2_hysteresis():
    print("\nHYSTERESIS")
    Path("rg/results/sage_corrected").mkdir(parents=True, exist_ok=True)
    
    omega_current = OMEGA_0.copy()
    forward_norms = []
    for eta in ETA_HYSTERESIS:
        trajectory = odeint(master_equation_sage, omega_current, T_SPAN, args=(eta, LAMBDA_FIXED, GAMMA_FIXED, I_EFF, ALPHA, BETA, SKEW, K))
        omega_current = trajectory[-1]
        forward_norms.append(np.linalg.norm(omega_current))
    
    omega_current = forward_norms[-1] * OMEGA_0 / np.linalg.norm(OMEGA_0)
    backward_norms = []
    for eta in reversed(ETA_HYSTERESIS):
        trajectory = odeint(master_equation_sage, omega_current, T_SPAN, args=(eta, LAMBDA_FIXED, GAMMA_FIXED, I_EFF, ALPHA, BETA, SKEW, K))
        omega_current = trajectory[-1]
        backward_norms.append(np.linalg.norm(omega_current))
    backward_norms = list(reversed(backward_norms))
    
    gap = np.max(np.abs(np.array(forward_norms) - np.array(backward_norms)))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ETA_HYSTERESIS, forward_norms, "b-o", label="Forward")
    ax.plot(ETA_HYSTERESIS, backward_norms, "r-s", label="Backward")
    ax.fill_between(ETA_HYSTERESIS, forward_norms, backward_norms, alpha=0.2)
    ax.set_xlabel("eta")
    ax.set_ylabel("||omega||")
    ax.set_title(f"Hysteresis (gap={gap:.3f})")
    ax.legend()
    plt.tight_layout()
    plt.savefig("rg/results/sage_corrected/hysteresis.png", dpi=300)
    plt.close()
    
    print(f"Max gap: {gap:.4f}")
    return gap > 0.01

def main():
    start = datetime.now()
    print("SAGE-CORRECTED VALIDATION")
    print(f"Start: {start}")
    
    grid, has_instability = task_1_phase_diagram()
    has_hysteresis = task_2_hysteresis()
    
    end = datetime.now()
    print(f"\nDuration: {end - start}")
    print(f"Instability: {has_instability}")
    print(f"Hysteresis: {has_hysteresis}")
    
    if has_instability and has_hysteresis:
        print("\nSUCCESS! Phase transition confirmed!")
    elif has_instability:
        print("\nPartial: instability found, weak hysteresis")
    else:
        print("\nNo instability detected")

if __name__ == "__main__":
    main()
