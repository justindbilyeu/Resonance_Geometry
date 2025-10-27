# Appendix D — DeepSeek Whitening Gain Derivation

## Motivation

When internal representations become ill-conditioned (high κ(Σ)), the system requires less external drive to destabilize. This appendix formalizes the adaptive coupling strength η_eff that accounts for covariance conditioning.

## Derivation

Consider the empirical covariance Σ over a sliding window of internal states. The condition number:

$$\kappa(\Sigma) = \frac{\lambda_{\max}}{\lambda_{\min}}$$

quantifies how ill-conditioned the representation is.

Define the **whitening gain**:

$$g_w = \frac{\log \kappa(\Sigma)}{d}$$

where d is the state dimensionality. This gain amplifies resonance when κ >> 1.

The **effective coupling strength** becomes:

$$\eta_{\mathrm{eff}} = \eta \cdot \left(1 + \tanh(g_w) \cdot \frac{c}{15}\right)$$

where c is a cap parameter (default 15.0) and tanh prevents runaway amplification.

## Phase Boundary Shift

The original boundary condition:

$$\eta \cdot \bar{I} \approx \lambda + \gamma$$

becomes:

$$\eta_{\mathrm{eff}} \cdot \bar{I} \approx \lambda + \gamma$$

Solving for the critical η:

$$\eta_{\mathrm{crit}} = \frac{\lambda + \gamma}{\bar{I} \cdot (1 + \tanh(g_w) \cdot c/15)}$$

**Key insight**: Higher conditioning (κ >> 1) **lowers** η_crit, shifting the boundary **leftward** in parameter space.

## Stabilization

To prevent numerical issues:
1. **Epsilon regularization**: Σ → Σ + ε·I (ε = 1e-12)
2. **Log clamping**: κ ∈ [1, 10¹²]
3. **Tanh capping**: Bounds gain contribution
4. **EMA smoothing**: α = 0.1 to prevent step jitter

## Implementation

See `src/resonance_geometry/hallucination/adaptive_gain.py`:
- `compute_effective_eta()` - Core computation
- `EtaEffEMA` - Exponential moving average smoother

## Attribution

Derivation: DeepSeek (conversation contribution, 2025)
Integration: Sage + xAI/Grok (validation)
