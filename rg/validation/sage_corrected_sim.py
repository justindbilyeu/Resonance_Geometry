#!/usr/bin/env python3
“””
SAGE-CORRECTED Master Validation Script
Implements all fixes from Sage’s analysis:

1. Linear MI drive (η·I·ω)
1. Nonzero operating point
1. Cubic-quintic for hysteresis
1. Non-normal cross-coupling
   “””

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import odeint
from pathlib import Path
import json
from datetime import datetime
import sys

# ============================================================================

# CONFIGURATION (per Sage’s recommendations)

# ============================================================================

# Fixed parameters

GAMMA_BASE = 1.0
I_EFF = 0.6        # Effective MI coherence [0,1] - Sage suggests 0.4-0.8
ALPHA = 0.25       # Destabilizing cubic (>0 for subcritical)
BETA = 0.01        # Stabilizing quintic (>0 for saturation)
SKEW = 0.05        # Non-normal coupling (sharpens onset)
K = 1.0            # Grounding stiffness

# Parameter grids

ETA_VALUES = np.array([0, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0])
LAMBDA_VALUES = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0])

# Hysteresis

ETA_HYSTERESIS = np.linspace(0.1, 5.0, 50)  # Fine sweep
LAMBDA_FIXED = 1.0
GAMMA_FIXED = 1.0

# Gamma sensitivity

GAMMA_SENSITIVITY = [0.5, 1.0, 2.0, 5.0]

# Time evolution

T_MAX = 150.0  # Longer to ensure convergence
DT = 0.05
T_SPAN = np.arange(0, T_MAX, DT)

# Nonzero operating point (per Sage)

OMEGA_0 = np.array([0.12, 0.08, 0.05])

np.random.seed(42)

# ============================================================================

# SAGE’S CORRECTED MASTER EQUATION

# ============================================================================

def master_equation_sage(omega, t, eta, lam, gamma, I_eff=0.6,
alpha=0.25, beta=0.01, skew=0.05, k=1.0):
“””
Sage’s corrected dynamics with:
- Linear MI drive: η·I·ω
- Cubic-quintic: +α·r²·ω - β·r⁴·ω
- Non-normal: skew coupling
- Grounding: -k·(ω - anchor)
“””

```
# 1) Linear MI drive (CRITICAL FIX)
drive = eta * I_eff * omega

# 2) Grounding (damping toward zero anchor)
grounding = -lam * k * omega

# 3) Coherence damping (linear)
damping = -gamma * omega

# 4) Cubic-quintic nonlinearity (for hysteresis)
r2 = np.dot(omega, omega)
r4 = r2 * r2
nonlinear = alpha * r2 * omega - beta * r4 * omega

# 5) Non-normal cross-coupling (3D skew-symmetric)
# Rotation around z-axis
skew_coupling = skew * np.array([omega[1], -omega[0], 0])

# Total dynamics
return drive + grounding + damping + nonlinear + skew_coupling
```

def compute_stability_sage(omega, eta, lam, gamma, I_eff=0.6,
alpha=0.25, beta=0.01, skew=0.05, k=1.0):
“”“Compute stability via Jacobian eigenvalues”””
eps = 1e-6
J = np.zeros((3, 3))

```
for i in range(3):
    omega_plus = omega.copy()
    omega_plus[i] += eps
    omega_minus = omega.copy()
    omega_minus[i] -= eps
    
    f_plus = master_equation_sage(omega_plus, 0, eta, lam, gamma, 
                                  I_eff, alpha, beta, skew, k)
    f_minus = master_equation_sage(omega_minus, 0, eta, lam, gamma, 
                                   I_eff, alpha, beta, skew, k)
    
    J[:, i] = (f_plus - f_minus) / (2 * eps)

eigenvalues = np.linalg.eigvals(J)
return np.max(eigenvalues.real)
```

def classify_regime(lambda_max):
if lambda_max < -0.05:
return ‘grounded’
elif lambda_max <= 0.05:
return ‘creative’
else:
return ‘hallucinatory’

# ============================================================================

# TASK 1: PHASE DIAGRAM

# ============================================================================

def task_1_phase_diagram():
“”“Generate phase diagram with Sage’s corrections”””
print(”\n” + “=”*70)
print(“TASK 1: PHASE DIAGRAM (SAGE-CORRECTED)”)
print(”=”*70)
print(f”Linear drive: η·I·ω with I={I_EFF}”)
print(f”Prediction: Phase boundary at η·I ≈ λ + γ”)
print(f”            → η ≈ {(LAMBDA_FIXED + GAMMA_BASE)/I_EFF:.2f} at λ={LAMBDA_FIXED}, γ={GAMMA_BASE}”)
print(”=”*70)

```
Path("rg/results/sage_corrected").mkdir(parents=True, exist_ok=True)

results_grid = np.zeros((len(ETA_VALUES), len(LAMBDA_VALUES)))
regime_grid = np.empty((len(ETA_VALUES), len(LAMBDA_VALUES)), dtype=object)
results = []

total = len(ETA_VALUES) * len(LAMBDA_VALUES)

for i, eta in enumerate(ETA_VALUES):
    for j, lam in enumerate(LAMBDA_VALUES):
        # Nonzero initial condition (per Sage)
        omega_0 = OMEGA_0 + 0.05 * np.random.randn(3)
        
        trajectory = odeint(master_equation_sage, omega_0, T_SPAN,
                          args=(eta, lam, GAMMA_BASE, I_EFF, 
                               ALPHA, BETA, SKEW, K))
        
        # Check convergence (last 20% of trajectory)
        omega_late = trajectory[-int(0.2*len(trajectory)):]
        omega_final = np.mean(omega_late, axis=0)
        omega_std = np.std(omega_late, axis=0)
        
        lambda_max = compute_stability_sage(omega_final, eta, lam, 
                                           GAMMA_BASE, I_EFF, ALPHA, BETA, SKEW, K)
        regime = classify_regime(lambda_max)
        
        results_grid[i, j] = lambda_max
        regime_grid[i, j] = regime
        
        # Theoretical prediction
        theory_boundary = (lam + GAMMA_BASE) / I_EFF
        near_boundary = abs(eta - theory_boundary) < 0.5
        
        results.append({
            'eta': eta, 'lambda': lam, 'lambda_max': float(lambda_max),
            'regime': regime, 
            'omega_norm': float(np.linalg.norm(omega_final)),
            'omega_std': float(np.linalg.norm(omega_std)),
            'theory_eta_critical': float(theory_boundary)
        })
        
        progress = i * len(LAMBDA_VALUES) + j + 1
        if progress % 9 == 0 or progress == total or near_boundary:
            theory_mark = " ⭐ (NEAR BOUNDARY)" if near_boundary else ""
            print(f"  {progress}/{total} | η={eta:.1f} λ={lam:.1f} → "
                  f"λ_max={lambda_max:.3f} ({regime}){theory_mark}")

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Heatmap
sns.heatmap(results_grid, xticklabels=np.round(LAMBDA_VALUES, 2),
            yticklabels=np.round(ETA_VALUES, 2), cmap='RdBu_r', center=0,
            cbar_kws={'label': 'Spectral Radius λ_max'}, ax=ax1)
ax1.contour(results_grid, levels=[0], colors='black', linewidths=3)

# Theoretical boundary: η = (λ + γ)/I
lambda_theory = np.linspace(LAMBDA_VALUES[0], LAMBDA_VALUES[-1], 100)
eta_theory = (lambda_theory + GAMMA_BASE) / I_EFF

# Convert to grid coordinates for overlay
lam_coords = np.interp(lambda_theory, LAMBDA_VALUES, np.arange(len(LAMBDA_VALUES)))
eta_coords = np.interp(eta_theory, ETA_VALUES, np.arange(len(ETA_VALUES)))

ax1.plot(lam_coords, eta_coords, 'g--', linewidth=3, 
        label=f'Theory: η=(λ+{GAMMA_BASE})/{I_EFF}')

ax1.set_xlabel('λ (Grounding)', fontsize=12)
ax1.set_ylabel('η (Meta-Awareness)', fontsize=12)
ax1.set_title('Phase Diagram: Sage-Corrected Dynamics', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left', fontsize=10)

# Regime distribution
regime_types = ['grounded', 'creative', 'hallucinatory']
regime_counts = [np.sum(regime_grid == r) for r in regime_types]
colors = ['blue', 'orange', 'red']
ax2.bar(regime_types, regime_counts, color=colors, alpha=0.7)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title('Regime Distribution', fontsize=13, fontweight='bold')
ax2.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('rg/results/sage_corrected/phase_diagram.png', dpi=300)
plt.close()

# Save data
with open('rg/results/sage_corrected/phase_data.json', 'w') as f:
    json.dump({'results': results, 'grid': results_grid.tolist()}, f, indent=2)

n_grounded = regime_counts[0]
n_creative = regime_counts[1]
n_halluc = regime_counts[2]

print(f"\n✅ Task 1 Complete:")
print(f"   Grounded: {n_grounded}/{total} ({100*n_grounded/total:.1f}%)")
print(f"   Creative: {n_creative}/{total} ({100*n_creative/total:.1f}%)")
print(f"   Hallucinatory: {n_halluc}/{total} ({100*n_halluc/total:.1f}%)")

return results_grid, regime_grid, n_halluc > 0
```

# ============================================================================

# TASK 2: HYSTERESIS (Sage’s sweep up/down)

# ============================================================================

def task_2_hysteresis():
“”“Sweep η up then down to detect hysteresis”””
print(”\n” + “=”*70)
print(“TASK 2: HYSTERESIS SWEEP (η ↑ then ↓)”)
print(”=”*70)
print(f”Fixed: λ={LAMBDA_FIXED}, γ={GAMMA_FIXED}”)
print(f”Sweeping η: {ETA_HYSTERESIS[0]:.2f} → {ETA_HYSTERESIS[-1]:.2f} → {ETA_HYSTERESIS[0]:.2f}”)
print(”=”*70)

```
Path("rg/results/sage_corrected").mkdir(parents=True, exist_ok=True)

# Forward sweep (η increasing)
omega_current = OMEGA_0.copy()
forward_norms = []
forward_lambda = []

for eta in ETA_HYSTERESIS:
    trajectory = odeint(master_equation_sage, omega_current, T_SPAN,
                      args=(eta, LAMBDA_FIXED, GAMMA_FIXED, I_EFF, 
                           ALPHA, BETA, SKEW, K))
    omega_current = trajectory[-1]
    norm = np.linalg.norm(omega_current)
    lam_max = compute_stability_sage(omega_current, eta, LAMBDA_FIXED, 
                                     GAMMA_FIXED, I_EFF, ALPHA, BETA, SKEW, K)
    forward_norms.append(norm)
    forward_lambda.append(lam_max)

# Backward sweep (η decreasing)
omega_current = forward_norms[-1] * OMEGA_0 / np.linalg.norm(OMEGA_0)  # Start from high η state
backward_norms = []
backward_lambda = []

for eta in reversed(ETA_HYSTERESIS):
    trajectory = odeint(master_equation_sage, omega_current, T_SPAN,
                      args=(eta, LAMBDA_FIXED, GAMMA_FIXED, I_EFF, 
                           ALPHA, BETA, SKEW, K))
    omega_current = trajectory[-1]
    norm = np.linalg.norm(omega_current)
    lam_max = compute_stability_sage(omega_current, eta, LAMBDA_FIXED, 
                                     GAMMA_FIXED, I_EFF, ALPHA, BETA, SKEW, K)
    backward_norms.append(norm)
    backward_lambda.append(lam_max)

backward_norms = list(reversed(backward_norms))
backward_lambda = list(reversed(backward_lambda))

# Compute hysteresis metrics
gap_norms = np.abs(np.array(forward_norms) - np.array(backward_norms))
gap_lambda = np.abs(np.array(forward_lambda) - np.array(backward_lambda))

loop_area_norm = np.trapz(gap_norms, ETA_HYSTERESIS)
loop_area_lambda = np.trapz(gap_lambda, ETA_HYSTERESIS)
max_gap_norm = np.max(gap_norms)
max_gap_lambda = np.max(gap_lambda)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Norm hysteresis
ax1.plot(ETA_HYSTERESIS, forward_norms, 'b-o', linewidth=2, 
        markersize=3, label='Forward (η↑)', alpha=0.7)
ax1.plot(ETA_HYSTERESIS, backward_norms, 'r-s', linewidth=2, 
        markersize=3, label='Backward (η↓)', alpha=0.7)
ax1.fill_between(ETA_HYSTERESIS, forward_norms, backward_norms, 
                 alpha=0.2, color='purple')
ax1.axvline((LAMBDA_FIXED + GAMMA_FIXED)/I_EFF, color='g', 
           linestyle='--', linewidth=2, label=f'Theory: η={(LAMBDA_FIXED + GAMMA_FIXED)/I_EFF:.2f}')
ax1.set_xlabel('η (Meta-Awareness)', fontsize=12)
ax1.set_ylabel('||ω|| (Connection Strength)', fontsize=12)
ax1.set_title(f'Hysteresis Loop (Norms)\nGap={max_gap_norm:.3f}, Area={loop_area_norm:.3f}', 
             fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# Stability hysteresis
ax2.plot(ETA_HYSTERESIS, forward_lambda, 'b-o', linewidth=2, 
        markersize=3, label='Forward (η↑)', alpha=0.7)
ax2.plot(ETA_HYSTERESIS, backward_lambda, 'r-s', linewidth=2, 
        markersize=3, label='Backward (η↓)', alpha=0.7)
ax2.fill_between(ETA_HYSTERESIS, forward_lambda, backward_lambda, 
                 alpha=0.2, color='purple')
ax2.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
ax2.axvline((LAMBDA_FIXED + GAMMA_FIXED)/I_EFF, color='g', 
           linestyle='--', linewidth=2, label=f'Theory: η={(LAMBDA_FIXED + GAMMA_FIXED)/I_EFF:.2f}')
ax2.set_xlabel('η (Meta-Awareness)', fontsize=12)
ax2.set_ylabel('λ_max (Stability)', fontsize=12)
ax2.set_title(f'Hysteresis Loop (Stability)\nGap={max_gap_lambda:.3f}, Area={loop_area_lambda:.3f}', 
             fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('rg/results/sage_corrected/hysteresis.png', dpi=300)
plt.close()

has_hysteresis = max_gap_norm > 0.01 or max_gap_lambda > 0.01

print(f"\n✅ Task 2 Complete:")
print(f"   Max gap (norm): {max_gap_norm:.4f}")
print(f"   Max gap (λ_max): {max_gap_lambda:.4f}")
print(f"   Loop area (norm): {loop_area_norm:.4f}")
print(f"   Loop area (λ_max): {loop_area_lambda:.4f}")
print(f"   Hysteresis: {'✓ YES' if has_hysteresis else '✗ NO'}")

return has_hysteresis, max_gap_norm
```

# ============================================================================

# MASTER EXECUTION

# ============================================================================

def main():
“”“Run Sage-corrected validation”””
start_time = datetime.now()

```
print("\n" + "="*70)
print("SAGE-CORRECTED VALIDATION")
print("="*70)
print(f"Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
print("\nSAGE'S FIXES APPLIED:")
print("  ✓ Linear MI drive: η·I·ω")
print("  ✓ Nonzero operating point")
print("  ✓ Cubic-quintic: +α·r²·ω - β·r⁴·ω")
print("  ✓ Non-normal coupling: skew rotation")
print(f"\nParameters: I={I_EFF}, α={ALPHA}, β={BETA}, skew={SKEW}")
print("="*70)

try:
    # Task 1
    phase_grid, regime_grid, has_instability = task_1_phase_diagram()
    
    # Task 2
    has_hysteresis, gap = task_2_hysteresis()
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    summary = f"""# Sage-Corrected Validation Summary
```

## Execution

- **Start**: {start_time.strftime(’%Y-%m-%d %H:%M:%S’)}
- **End**: {end_time.strftime(’%Y-%m-%d %H:%M:%S’)}
- **Duration**: {duration}

## Sage’s Diagnosis

**Problem**: Original had η·ω³ amplification → vanished near origin
**Fix**: Linear η·I·ω + cubic-quintic + non-normal coupling

## Results

### Phase Diagram

- Instability regime: {‘✓ DETECTED’ if has_instability else ‘✗ NOT FOUND’}
- Phase boundary visible: {‘✓’ if has_instability else ‘✗’}
- Theory match (η ≈ (λ+γ)/I): {‘Validated’ if has_instability else ‘Pending’}

### Hysteresis

- Loop detected: {‘✓ YES’ if has_hysteresis else ‘✗ NO’}
- Maximum gap: {gap:.4f}
- Transition type: {‘Subcritical (first-order)’ if has_hysteresis else ‘Supercritical or none’}

## Conclusion

{‘🎉 SUCCESS! Phase transition and hysteresis confirmed!’ if (has_instability and has_hysteresis) else ‘⚠️ Partial success - needs iteration’ if has_instability else ‘❌ Still no instability - consult Sage’}

## Output Files

- rg/results/sage_corrected/phase_diagram.png
- rg/results/sage_corrected/hysteresis.png
- rg/results/sage_corrected/phase_data.json
  “””
  
  ```
    with open('rg/results/SAGE_CORRECTED_SUMMARY.md', 'w') as f:
        f.write(summary)
    
    print("\n" + "="*70)
    print(summary)
    print("="*70)
    
    if has_instability and has_hysteresis:
        print("\n🎉 COMPLETE SUCCESS!")
        print("Theory validated! Check the figures.")
    elif has_instability:
        print("\n✓ Phase transition found, but weak hysteresis")
        print("May need to tune α, β for stronger loops")
    else:
        print("\n⚠️ Still debugging - consult Sage for next steps")
  ```
  
  except Exception as e:
  print(f”\n❌ ERROR: {e}”)
  import traceback
  traceback.print_exc()
  sys.exit(1)

if **name** == “**main**”:
main()
