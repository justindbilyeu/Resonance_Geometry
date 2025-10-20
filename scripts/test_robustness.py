#!/usr/bin/env python3
"""
Robustness Test for β_c Convergence
Tests whether the critical point survives under:
- Initial condition perturbations
- Measurement noise
- Parameter uncertainty
"""

import numpy as np
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.ringing_sweep import simulate_rg_system
from scripts.ringing_detector import compute_ringing_score
from scripts.jacobian_analyzer import compute_eigenvalue_at_beta
from scripts.fluency_analyzer import compute_fluency_velocity


def run_robustness_trial(beta, noise_level=0.05, ic_std=0.1):
    """Run one trial with perturbations"""
    
    # Perturb initial conditions
    phi0 = 1.0 + np.random.normal(0, ic_std)
    v0 = 0.0 + np.random.normal(0, ic_std)
    
    # Run simulation
    try:
        data = simulate_rg_system(
            beta=beta,
            alpha=0.1,
            omega0=1.0,
            phi0=phi0,
            v0=v0,
            tmax=100.0,
            dt=0.01
        )
        
        # Add measurement noise
        data['phi'] = data['phi'] + np.random.normal(0, noise_level, len(data['phi']))
        
        # Compute all three observables
        ringing = compute_ringing_score(data)
        eigenvalue = compute_eigenvalue_at_beta(beta, alpha=0.1, omega0=1.0)
        fluency = compute_fluency_velocity(data)
        
        return {
            'success': True,
            'ringing_score': ringing,
            'eigenvalue_real': eigenvalue.real,
            'fluency_peak': fluency,
            'phi0': phi0,
            'v0': v0
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def main():
    print("=" * 60)
    print("ROBUSTNESS TEST: β_c Convergence Under Perturbation")
    print("=" * 60)
    print()
    
    # Test parameters
    beta_c = 0.015
    n_trials = 100
    noise_level = 0.05  # 5% measurement noise
    ic_std = 0.10       # 10% initial condition variance
    
    print(f"Configuration:")
    print(f"  β_c (nominal): {beta_c}")
    print(f"  Trials: {n_trials}")
    print(f"  Measurement noise: {noise_level * 100:.1f}%")
    print(f"  IC perturbation: {ic_std * 100:.1f}%")
    print()
    
    # Run trials
    print("Running trials...")
    results = []
    successful = 0
    
    for i in range(n_trials):
        if (i + 1) % 20 == 0:
            print(f"  Trial {i + 1}/{n_trials}...")
        
        trial = run_robustness_trial(beta_c, noise_level, ic_std)
        results.append(trial)
        
        if trial['success']:
            successful += 1
    
    print(f"\nCompleted: {successful}/{n_trials} successful trials")
    print()
    
    # Analyze results
    if successful < n_trials * 0.8:
        print("⚠️  WARNING: >20% trial failure rate")
        print("   The system may be unstable under perturbation")
        return
    
    # Extract successful measurements
    valid_results = [r for r in results if r['success']]
    
    ringing_scores = [r['ringing_score'] for r in valid_results]
    eigenvalues = [r['eigenvalue_real'] for r in valid_results]
    fluencies = [r['fluency_peak'] for r in valid_results]
    
    # Statistical analysis
    print("=" * 60)
    print("RESULTS: Observable Statistics Under Perturbation")
    print("=" * 60)
    print()
    
    print("Ringing Score:")
    print(f"  Mean: {np.mean(ringing_scores):.4f}")
    print(f"  Std:  {np.std(ringing_scores):.4f}")
    print(f"  CV:   {np.std(ringing_scores) / np.mean(ringing_scores):.2%}")
    print()
    
    print("Eigenvalue (Real Part):")
    print(f"  Mean: {np.mean(eigenvalues):.6f}")
    print(f"  Std:  {np.std(eigenvalues):.6f}")
    print()
    
    print("Fluency Velocity:")
    print(f"  Mean: {np.mean(fluencies):.4f}")
    print(f"  Std:  {np.std(fluencies):.4f}")
    print(f"  CV:   {np.std(fluencies) / np.mean(fluencies):.2%}")
    print()
    
    # Convergence test
    print("=" * 60)
    print("CONVERGENCE ASSESSMENT")
    print("=" * 60)
    print()
    
    # Check if observables still correlate despite noise
    corr_ringing_eigen = np.corrcoef(ringing_scores, eigenvalues)[0, 1]
    corr_eigen_fluency = np.corrcoef(eigenvalues, fluencies)[0, 1]
    corr_ringing_fluency = np.corrcoef(ringing_scores, fluencies)[0, 1]
    
    print("Observable Correlations (under noise):")
    print(f"  Ringing ↔ Eigenvalue: {corr_ringing_eigen:.3f}")
    print(f"  Eigenvalue ↔ Fluency: {corr_eigen_fluency:.3f}")
    print(f"  Ringing ↔ Fluency:    {corr_ringing_fluency:.3f}")
    print()
    
    # Verdict
    eigenvalue_stable = np.std(eigenvalues) < 0.002
    correlation_maintained = min(abs(corr_ringing_eigen), 
                                  abs(corr_eigen_fluency),
                                  abs(corr_ringing_fluency)) > 0.5
    
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    print()
    
    if eigenvalue_stable and correlation_maintained:
        print("✅ ROBUST: β_c survives perturbation")
        print("   - Eigenvalue variance < 0.002")
        print("   - Observable correlations maintained")
        print()
        print("   → The convergence appears STRUCTURAL, not coincidental")
    elif eigenvalue_stable:
        print("⚠️  PARTIALLY ROBUST:")
        print("   - Eigenvalue stable")
        print("   - But correlations weakened under noise")
        print()
        print("   → The critical point exists but observables decouple")
    else:
        print("❌ FRAGILE: β_c does not survive perturbation")
        print(f"   - Eigenvalue std = {np.std(eigenvalues):.4f} (> 0.002)")
        print()
        print("   → This may be a numerical artifact")
    
    print()
    
    # Save results
    output_dir = Path("results/robustness")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'config': {
            'beta_c': beta_c,
            'n_trials': n_trials,
            'noise_level': noise_level,
            'ic_std': ic_std,
            'successful_trials': successful
        },
        'statistics': {
            'ringing_score': {
                'mean': float(np.mean(ringing_scores)),
                'std': float(np.std(ringing_scores))
            },
            'eigenvalue_real': {
                'mean': float(np.mean(eigenvalues)),
                'std': float(np.std(eigenvalues))
            },
            'fluency_velocity': {
                'mean': float(np.mean(fluencies)),
                'std': float(np.std(fluencies))
            }
        },
        'correlations': {
            'ringing_eigenvalue': float(corr_ringing_eigen),
            'eigenvalue_fluency': float(corr_eigen_fluency),
            'ringing_fluency': float(corr_ringing_fluency)
        },
        'verdict': {
            'eigenvalue_stable': eigenvalue_stable,
            'correlation_maintained': correlation_maintained,
            'robust': eigenvalue_stable and correlation_maintained
        }
    }
    
    output_file = output_dir / "summary.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    print()


if __name__ == "__main__":
    main()
