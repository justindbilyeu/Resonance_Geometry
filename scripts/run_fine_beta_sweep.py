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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Optional plotting
try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False


def main():
    print("=" * 70)
    print("FINE β SWEEP: Locating Critical Point")
    print("=" * 70)
    print()
    
    # Import required modules
    try:
        from experiments.gp_ringing_demo import simulate_coupled, vector_field
        from experiments.jacobian import finite_diff_jacobian, max_real_eig
        from experiments.ringing_detector import detect_ringing
    except ImportError as e:
        print(f"ERROR: Missing required module: {e}")
        print("\nRequired files:")
        print("  - experiments/gp_ringing_demo.py")
        print("  - experiments/jacobian.py")
        print("  - experiments/ringing_detector.py")
        sys.exit(1)
    
    # Fixed parameters
    alpha = 0.1
    eta = 0.0
    tau = 10.0
    K0 = 0.1
    
    # Fine β sweep
    betas = np.linspace(0.010, 0.050, 21)
    
    print(f"Configuration:")
    print(f"  α (alpha):  {alpha}")
    print(f"  τ (tau):    {tau}")
    print(f"  K₀:         {K0}")
    print(f"  η (noise):  {eta}")
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
            'eta': eta,
            'tau': tau,
            'K0': K0
        }
        
        # 1. Run simulation
        try:
            t, states = simulate_coupled(
                steps=600,
                seed=42,
                **params
            )
            series = states.mean(axis=1) if getattr(states, 'ndim', 1) > 1 else states
        except Exception as e:
            print(f"  WARNING: Simulation failed at β={beta:.4f}: {e}")
            continue
        
        # 2. Detect ringing
        ringing_result = detect_ringing(series)
        
        # 3. Compute Jacobian eigenvalue
        try:
            # Use final state as equilibrium approximation
            q = np.full(4, float(series[-1]))
            p = np.zeros_like(q)
            x0 = np.concatenate([q, p])
            
            J = finite_diff_jacobian(vector_field, x0, params)
            max_eig = max_real_eig(J)
        except Exception as e:
            print(f"  WARNING: Jacobian failed at β={beta:.4f}: {e}")
            max_eig = None
        
        # Store results
        result = {
            'beta': float(beta),
            'ringing': ringing_result['ringing'],
            'ringing_score': ringing_result.get('score', 0.0),
            'max_real_eig': float(max_eig) if max_eig is not None else None,
            'series_mean': float(np.mean(series)),
            'series_std': float(np.std(series))
        }
        results.append(result)
    
    print(f"\nCompleted: {len(results)}/{len(betas)} successful")
    print()
    
    # Analysis
    print("=" * 70)
    print("ANALYSIS: Finding Critical Point")
    print("=" * 70)
    print()
    
    # Extract arrays
    beta_vals = np.array([r['beta'] for r in results])
    ringing_vals = np.array([r['ringing'] for r in results])
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
        print("  No clear ringing transition found")
        beta_ringing_start = None
    
    print()
    
    # Find eigenvalue crossing
    print("Eigenvalue Zero-Crossing:")
    if len(eig_vals) > 0:
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
            print("   May indicate:")
            print("   - Need finer resolution")
            print("   - Different transition mechanisms")
            print("   - Measurement sensitivity issues")
            beta_c = None
    else:
        print("❌ INSUFFICIENT DATA: Cannot assess convergence")
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
            'K0': K0,
            'eta': eta,
            'beta_range': [float(betas[0]), float(betas[-1])],
            'n_points': len(betas)
        },
        'transitions': {
            'ringing_onset': float(beta_ringing_start) if beta_ringing_start is not None else None,
            'eigenvalue_crossing': float(beta_eig_cross) if beta_eig_cross is not None else None,
            'difference': float(abs(beta_ringing_start - beta_eig_cross)) if (beta_ringing_start and beta_eig_cross) else None
        },
        'beta_c': float(beta_c) if beta_c is not None else None,
        'convergence': beta_c is not None and abs(beta_ringing_start - beta_eig_cross) < 0.005 if (beta_ringing_start and beta_eig_cross) else False
    }
    
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Results saved to: {output_dir}/")
    print(f"  - sweep_data.json (raw data)")
    print(f"  - summary.json (analysis)")
    print()
    
    # Plot if available
    if HAS_PLT and len(results) > 0:
        print("Generating plots...")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot 1: Ringing detection
        ax1.scatter(beta_vals[ringing_vals], 
                   np.ones(np.sum(ringing_vals)), 
                   c='red', marker='o', s=100, label='Ringing', zorder=3)
        ax1.scatter(beta_vals[~ringing_vals], 
                   np.zeros(np.sum(~ringing_vals)), 
                   c='blue', marker='x', s=100, label='No ringing', zorder=3)
        ax1.set_ylabel('Ringing Detected')
        ax1.set_ylim(-0.2, 1.2)
        ax1.set_yticks([0, 1])
        ax1.set_yticklabels(['No', 'Yes'])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Fine β Sweep: Critical Point Detection')
        
        if beta_ringing_start is not None:
            ax1.axvline(beta_ringing_start, color='red', linestyle='--', 
                       alpha=0.5, label=f'Onset: {beta_ringing_start:.4f}')
        
        # Plot 2: Eigenvalue
        if len(eig_vals) > 0:
            ax2.plot(eig_betas, eig_vals, 'o-', color='green', linewidth=2, markersize=6)
            ax2.axhline(0, color='black', linestyle='-', linewidth=1)
            ax2.set_ylabel('Max Re(λ)')
            ax2.set_xlabel('β')
            ax2.grid(True, alpha=0.3)
            
            if beta_eig_cross is not None:
                ax2.axvline(beta_eig_cross, color='green', linestyle='--', 
                           alpha=0.5, label=f'Crossing: {beta_eig_cross:.4f}')
                ax2.legend()
        
        plt.tight_layout()
        
        fig_file = output_dir / "beta_sweep_analysis.png"
        plt.savefig(fig_file, dpi=150, bbox_inches='tight')
        print(f"  - beta_sweep_analysis.png (visualization)")
        plt.close()
        
        # Also save to docs
        docs_dir = Path("docs/assets/figures")
        docs_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 8))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        ax1.scatter(beta_vals[ringing_vals], 
                   np.ones(np.sum(ringing_vals)), 
                   c='red', marker='o', s=100, label='Ringing', zorder=3)
        ax1.scatter(beta_vals[~ringing_vals], 
                   np.zeros(np.sum(~ringing_vals)), 
                   c='blue', marker='x', s=100, label='No ringing', zorder=3)
        ax1.set_ylabel('Ringing Detected')
        ax1.set_ylim(-0.2, 1.2)
        ax1.set_yticks([0, 1])
        ax1.set_yticklabels(['No', 'Yes'])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Critical Point β_c: Ringing + Eigenvalue Convergence')
        
        if beta_ringing_start is not None:
            ax1.axvline(beta_ringing_start, color='red', linestyle='--', alpha=0.5)
        
        if len(eig_vals) > 0:
            ax2.plot(eig_betas, eig_vals, 'o-', color='green', linewidth=2, markersize=6)
            ax2.axhline(0, color='black', linestyle='-', linewidth=1)
            ax2.set_ylabel('Max Re(λ)')
            ax2.set_xlabel('β')
            ax2.grid(True, alpha=0.3)
            
            if beta_eig_cross is not None:
                ax2.axvline(beta_eig_cross, color='green', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(docs_dir / "beta_c_convergence.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  - docs/assets/figures/beta_c_convergence.png (for wiki)")
    
    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
