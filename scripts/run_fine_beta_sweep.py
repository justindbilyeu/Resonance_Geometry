#!/usr/bin/env python3
"""
Fine-resolution β sweep to locate critical point.
Tests β ∈ [0.010, 0.050] with Jacobian eigenvalues and ringing detection.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import odeint

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Optional plotting
try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False


def rg_dynamics(state, t, beta, alpha, omega0, K0):
    """
    Resonance Geometry dynamics matching vector_field structure.
    
    This is the actual system being tested:
    dphi/dt = v
    dv/dt = -beta*v - omega0^2*phi + K0*sin(alpha*phi)
    """
    phi, v = state
    dphi = v
    dv = -beta * v - (omega0**2) * phi + K0 * np.sin(alpha * phi)
    return [dphi, dv]


def main():
    print("=" * 70)
    print("FINE β SWEEP: Locating Critical Point")
    print("=" * 70)
    print()
    
    # Import required modules
    try:
        from experiments.gp_ringing_demo import vector_field
        from experiments.jacobian import finite_diff_jacobian, max_real_eig
        from experiments.ringing_detector import detect_ringing
    except ImportError as e:
        print(f"ERROR: Missing required module: {e}")
        sys.exit(1)
    
    # Fixed parameters
    alpha = 0.1
    tau = 10.0
    K0 = 0.1
    omega0 = 1.0 / tau
    
    # Fine β sweep
    betas = np.linspace(0.010, 0.050, 21)
    
    print(f"Configuration:")
    print(f"  α (alpha):  {alpha}")
    print(f"  τ (tau):    {tau}")
    print(f"  ω₀:         {omega0:.3f}")
    print(f"  K₀:         {K0}")
    print(f"  β range:    {betas[0]:.3f} to {betas[-1]:.3f}")
    print(f"  β steps:    {len(betas)}")
    print()
    
    # Run sweep
    print("Running simulations...")
    results = []
    
    for i, beta in enumerate(betas):
        if (i + 1) % 5 == 0:
            print(f"  β = {beta:.4f} ({i+1}/{len(betas)})")
        
        params = {
            'alpha': alpha,
            'beta': beta,
            'tau': tau,
            'K0': K0
        }
        
        # 1. Run simulation
        try:
            t_span = np.linspace(0, 100, 1000)  # 100 time units, 1000 points
            initial = [1.0, 0.0]  # phi0=1.0, v0=0.0
            
            sol = odeint(rg_dynamics, initial, t_span, args=(beta, alpha, omega0, K0))
            series = sol[:, 0]  # phi(t)
            
        except Exception as e:
            print(f"  WARNING: Simulation failed at β={beta:.4f}: {e}")
            continue
        
        # 2. Detect ringing
        try:
            ringing_result = detect_ringing(series)
        except Exception as e:
            print(f"  WARNING: Ringing detection failed at β={beta:.4f}: {e}")
            ringing_result = {'ringing': False, 'score': 0.0}
        
        # 3. Compute Jacobian eigenvalue at equilibrium (phi=0, v=0)
        try:
            # Equilibrium is at phi=0 for this system
            q = np.zeros(4)  # 4 oscillators at equilibrium
            p = np.zeros(4)
            x0 = np.concatenate([q, p])
            
            J = finite_diff_jacobian(vector_field, x0, params)
            max_eig = max_real_eig(J)
        except Exception as e:
            print(f"  WARNING: Jacobian failed at β={beta:.4f}: {e}")
            max_eig = None
        
        # Store results
        result_dict = {
            'beta': float(beta),
            'ringing': bool(ringing_result.get('ringing', False)),
            'ringing_score': float(ringing_result.get('score', 0.0)),
            'max_real_eig': float(max_eig) if max_eig is not None else None,
            'series_mean': float(np.mean(series)),
            'series_std': float(np.std(series)),
            'series_final': float(series[-1])
        }
        results.append(result_dict)
    
    print(f"\nCompleted: {len(results)}/{len(betas)} successful")
    print()
    
    if len(results) == 0:
        print("ERROR: No successful simulations. Cannot proceed.")
        return
    
    # Analysis
    print("=" * 70)
    print("ANALYSIS: Finding Critical Point")
    print("=" * 70)
    print()
    
    # Extract arrays
    beta_vals = np.array([r['beta'] for r in results])
    ringing_vals = np.array([r['ringing'] for r in results])
    scores = np.array([r['ringing_score'] for r in results])
    eig_vals = np.array([r['max_real_eig'] for r in results if r['max_real_eig'] is not None])
    eig_betas = np.array([r['beta'] for r in results if r['max_real_eig'] is not None])
    
    # Find transitions
    print("Ringing Transition:")
    ringing_on = beta_vals[ringing_vals == True]
    ringing_off = beta_vals[ringing_vals == False]
    
    if len(ringing_on) > 0 and len(ringing_off) > 0:
        beta_ringing_start = np.min(ringing_on)
        beta_ringing_end = np.max(ringing_on)
        print(f"  First ringing:  β = {beta_ringing_start:.4f}")
        print(f"  Last ringing:   β = {beta_ringing_end:.4f}")
        print(f"  Ringing window: [{beta_ringing_start:.4f}, {beta_ringing_end:.4f}]")
    else:
        print(f"  Ringing detected: {np.sum(ringing_vals)}/{len(ringing_vals)} points")
        if np.sum(ringing_vals) == 0:
            print("  ⚠️  No ringing detected at any β value")
            print(f"  Score range: [{np.min(scores):.3f}, {np.max(scores):.3f}]")
        beta_ringing_start = None
    
    print()
    
    # Find eigenvalue crossing
    print("Eigenvalue Zero-Crossing:")
    if len(eig_vals) > 0:
        print(f"  Eigenvalue range: [{np.min(eig_vals):.6f}, {np.max(eig_vals):.6f}]")
        
        # Find where eigenvalue crosses zero
        sign_changes = np.where(np.diff(np.sign(eig_vals)))[0]
        
        if len(sign_changes) > 0:
            idx = sign_changes[0]
            beta_eig_cross = (eig_betas[idx] + eig_betas[idx + 1]) / 2
            print(f"  Zero crossing:  β ≈ {beta_eig_cross:.4f}")
            print(f"  Before: β={eig_betas[idx]:.4f}, λ={eig_vals[idx]:.6f}")
            print(f"  After:  β={eig_betas[idx+1]:.4f}, λ={eig_vals[idx+1]:.6f}")
        else:
            print("  No zero-crossing detected")
            beta_eig_cross = None
    else:
        print("  No eigenvalue data available")
        beta_eig_cross = None
    
    print()
    
    # Convergence assessment
    print("=" * 70)
    print("CONVERGENCE CHECK")
    print("=" * 70)
    print()
    
    if beta_ringing_start is not None and beta_eig_cross is not None:
        difference = abs(beta_ringing_start - beta_eig_cross)
        print(f"Ringing onset:       β = {beta_ringing_start:.4f}")
        print(f"Eigenvalue crossing: β = {beta_eig_cross:.4f}")
        print(f"Difference:          Δβ = {difference:.4f}")
        print()
        
        if difference < 0.005:
            print("✅ CONVERGENCE: Observables align within 0.005")
            print(f"   Critical point: β_c ≈ {(beta_ringing_start + beta_eig_cross)/2:.4f}")
            beta_c = (beta_ringing_start + beta_eig_cross) / 2
        else:
            print("⚠️  DIVERGENCE: Observables do not align closely")
            beta_c = None
    else:
        print("❌ INSUFFICIENT DATA: Cannot assess convergence")
        if beta_ringing_start is None:
            print("   - No ringing transition detected")
        if beta_eig_cross is None:
            print("   - No eigenvalue crossing detected")
        beta_c = None
    
    print()
    
    # Save results
    output_dir = Path("results/fine_beta_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw data
    data_file = output_dir / "sweep_data.json"
    with open(data_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save summary
    summary = {
        'config': {
            'alpha': alpha,
            'tau': tau,
            'omega0': omega0,
            'K0': K0,
            'beta_range': [float(betas[0]), float(betas[-1])],
            'n_points': len(betas)
        },
        'transitions': {
            'ringing_onset': float(beta_ringing_start) if beta_ringing_start is not None else None,
            'eigenvalue_crossing': float(beta_eig_cross) if beta_eig_cross is not None else None,
            'difference': float(abs(beta_ringing_start - beta_eig_cross)) if (beta_ringing_start and beta_eig_cross) else None
        },
        'beta_c': float(beta_c) if beta_c is not None else None,
        'convergence': bool(beta_c is not None) if beta_c is not None else False
    }
    
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Results saved to: {output_dir}/")
    print(f"  - sweep_data.json")
    print(f"  - summary.json")
    print()
    
    # Plotting
    if HAS_PLT and len(results) > 0:
        print("Generating plots...")
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        
        # Plot 1: Ringing detection
        if np.sum(ringing_vals) > 0:
            axes[0].scatter(beta_vals[ringing_vals], np.ones(np.sum(ringing_vals)), 
                           c='red', marker='o', s=100, label='Ringing', zorder=3)
        if np.sum(~ringing_vals) > 0:
            axes[0].scatter(beta_vals[~ringing_vals], np.zeros(np.sum(~ringing_vals)), 
                           c='blue', marker='x', s=100, label='No ringing', zorder=3)
        
        axes[0].set_ylabel('Ringing')
        axes[0].set_ylim(-0.2, 1.2)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('Fine β Sweep: Critical Point Search')
        
        # Plot 2: Ringing score
        axes[1].plot(beta_vals, scores, 'o-', color='orange', linewidth=2)
        axes[1].set_ylabel('Ringing Score')
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Eigenvalue
        if len(eig_vals) > 0:
            axes[2].plot(eig_betas, eig_vals, 'o-', color='green', linewidth=2, markersize=6)
            axes[2].axhline(0, color='black', linestyle='-', linewidth=1)
            axes[2].set_ylabel('Max Re(λ)')
            axes[2].grid(True, alpha=0.3)
            
            if beta_eig_cross is not None:
                for ax in axes:
                    ax.axvline(beta_eig_cross, color='green', linestyle='--', 
                              alpha=0.3, linewidth=2)
        
        if beta_ringing_start is not None:
            for ax in axes:
                ax.axvline(beta_ringing_start, color='red', linestyle='--', 
                          alpha=0.3, linewidth=2)
        
        axes[2].set_xlabel('β')
        
        plt.tight_layout()
        plt.savefig(output_dir / "beta_sweep_analysis.png", dpi=150, bbox_inches='tight')
        print(f"  - beta_sweep_analysis.png")
        plt.close()
    
    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
