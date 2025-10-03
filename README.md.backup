# Resonance Geometry

> **TL;DR**: What if networks could learn which connections matter by measuring how much information flows through them? We’re testing whether information itself can reshape structure—in software, neural networks, maybe even physical systems.

-----

## The Big Idea (In Plain English)

Imagine you’re at a party. Some conversations are memorable—you remember them, your friends remember them, other people overhear and remember them. Those conversations become stories that spread. Other chats are forgotten instantly.

**Resonance Geometry asks**: What if systems work the same way? What if the connections that carry useful information automatically get stronger, while useless connections fade?

We’re building math and simulations to test this. If it works, it might explain:

- Why some neural networks learn better than others
- How brains decide which connections to keep
- Whether information flow is a force that shapes structure (like gravity shapes orbits)

**Status**: Early research. We have working simulations and testable predictions. No revolutionary claims yet—just careful experiments.

-----

## Quick Navigation

- 🎯 **Just curious?** → Read [What We’re Testing](#what-were-testing) below
- 🚀 **Want to see it run?** → Check [Quick Demo](#quick-demo)
- 🔬 **Technical background?** → Jump to [For Researchers](#for-researchers)
- 💡 **Have questions?** → See [FAQ](#frequently-asked-questions)
- 🤝 **Want to contribute?** → Visit [Contributing](#contributing)

-----

## What We’re Testing

**Core Hypothesis**: Systems with feedback between information flow and connection strength will develop stable, efficient structures automatically.

**Concrete Predictions We’re Checking**:

1. **Ringing Boundary**: Systems should transition from smooth to oscillatory behavior at a mathematically predictable threshold
1. **Hysteresis Resonance**: Cyclic inputs should produce maximum response at specific frequencies
1. **Geometric Witness**: Information-rich connections should become structurally reinforced over time

**What Makes This Different**: We’re not philosophizing—we’re measuring. Every claim has a number attached and can be proven wrong.

### Example Results

```
Phase Boundary Map (placeholder - will show actual results)
Systems transition from stable → oscillatory exactly where equations predict

Hysteresis Curves (placeholder - will show actual results)  
Response peaks at resonant frequencies—information leaves geometric "memory"
```

-----

## Why This Might Matter

**If our predictions hold**:

- **AI/ML**: Better training algorithms that adapt network architecture automatically
- **Neuroscience**: Mathematical framework for how brains learn which connections matter
- **Complex Systems**: General principle for how information shapes structure

**If they don’t hold**:

- Still produces useful diagnostic tools for adaptive networks
- Clarifies what *doesn’t* work, which is also progress
- Demonstrates methodology for rigorous speculative science

**Either way, we learn something.**

-----

## Current Status

✅ **Working**: Math framework, simulation infrastructure, reproducible experiments  
🔄 **In Progress**: Phase 1 validation experiments, visualization pipeline  
🔴 **Blocked**: NetworkX integration (Task 0) - preventing key tests  
📊 **Results So Far**: Ringing boundary predictions confirmed in simplified models

**Honest Assessment**: Too early to claim success, but early results are encouraging enough to keep testing rigorously.

-----

## Quick Demo

Want to see it in action? If you have Python installed:

```bash
# Clone and set up (takes ~2 minutes)
git clone https://github.com/justindbilyeu/Resonance_Geometry.git
cd Resonance_Geometry
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run a quick experiment (~2 minutes)
python scripts/run_phase_sweep.py \
  --alphas "0.3,0.6" --etas "0.03,0.05" \
  --T 100 --M 10 --seed 42

# Results appear in results/phase/ as graphs and data
```

You’ll see phase maps showing where systems go unstable—exactly where our math predicts.

-----

## Why This Project Exists

Most research on complex systems treats structure and information separately:

- Engineers build network architectures, then see what information flows through them
- Neuroscientists map brain connections, then study what signals they carry

**We’re asking**: What if these aren’t separate? What if information flow *creates* structure, and structure *shapes* information flow, in an endless feedback loop?

It’s a simple question with potentially profound implications. We’re doing the math and experiments to find out if it’s true.

-----

## Frequently Asked Questions

**Q: Is this related to quantum mechanics?**  
A: Not directly. We use math from information theory and differential geometry, but it applies to any adaptive network—software, neural nets, social systems.

**Q: Could this explain consciousness?**  
A: We’re not making claims about consciousness. One axiom explores whether emotions might map to geometric curvature, but that’s highly speculative and clearly labeled as such in our documentation.

**Q: When will you know if this works?**  
A: Phase 1 experiments (currently in progress) will show whether our predictions match real system behavior. We expect initial results in 2-3 months.

**Q: Can I use this code for my project?**  
A: Yes! The code is open source. If it helps, great! But it’s research-grade—expect rough edges and changing APIs.

**Q: How can I help?**  
A: Test the code, suggest experiments, find bugs, ask good questions, or just spread the word. Open an issue on GitHub—we’re friendly!

**Q: Is this peer-reviewed?**  
A: Not yet. We’re in the experimental phase. Formal publication will come after Phase 1 validation is complete.

-----

## For Researchers

### Tagline

**How information flow sculpts structure.**

We study closed-loop dynamics where:

1. **Environments witness stable variables** (Resonant Witness Postulate - RWP)
1. **Systems adapt coupling geometry** to maximize useful records (Geometric Plasticity - GP)

This creates feedback: information flow reshapes structure, which reshapes information flow.

### Core Theoretical Framework

**Resonant Witness Postulate (RWP)**: Environments preferentially copy (“witness”) stable system variables, creating redundant records across space/time.

**Geometric Plasticity (GP)**: Couplings self-tune in proportion to the information they carry, closing a feedback loop between signal and structure.

**Key Innovation**: We’re testing whether this feedback produces:

- Predictable phase transitions (ringing boundaries)
- Resonant responses to periodic forcing (hysteresis peaks)
- Emergent structural motifs (broadcast ↔ modular architectures)

### What’s New

- **Ringing boundary**: Gain-controlled transition (smooth → underdamped) with closed-form Routh-Hurwitz threshold
- **Hysteresis resonance**: Loop area peaks at drive period matching natural system timescales
- **Motif universality**: Information-constrained systems converge to similar topologies
- **Engineering rule**: Practical $K_c$ threshold for stability in delayed feedback systems

### Repository Structure

```
docs/
├── whitepaper/              # Draft theoretical framework (GP/RWP)
├── appendices/              # Mathematical derivations
│   ├── appendix_ring_threshold.md      # Routh-Hurwitz stability
│   ├── appendix_hysteresis_prefactor.md # Resonance predictions
│   ├── appendix_motif_universality.md   # Topology emergence
│   └── appendix_delay_stability.md      # Delay effects
├── experiments/             # Protocol notes and methods
└── hardware/
    └── ITPU.md             # Information-Theoretic Processing Unit concept

src/                         # Core library
├── rwp_system.py           # System dynamics (S–F_k + plasticity)
├── plasticity.py           # GP update rules (EMA, Laplacian, budget)
├── metrics.py              # Mutual information, redundancy, witness flux
├── diagnostics.py          # PSD peaks, overshoots, damping ratios
└── utils.py                # Supporting functions

scripts/                     # Reproducible experiment runners
├── run_phase_sweep.py      # Ringing boundary (α × η grid)
├── run_hysteresis.py       # Loop area vs. period (ON/OFF forcing)
├── run_motif_sweep.py      # Broadcast ↔ modular topology sweep
└── run_phase_map_surrogate.py # AR(2) fast proxy for validation

theory/                      # Validation notebooks
├── kc_rule_validation.ipynb
├── hysteresis_fit.ipynb
└── identifiability_estimator.py

tests/                       # Unit & integration tests
results/                     # Generated data (CSVs/plots)
```

### Installation & Setup

```bash
# Create virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
# Includes: numpy, scipy, matplotlib, pandas, networkx, pytest
```

### Running Experiments

**Ringing Boundary (Full RWP System)**

```bash
python scripts/run_phase_sweep.py \
  --alphas "0.1,0.3,0.6,0.9" \
  --etas "0.01,0.03,0.05,0.08" \
  --T 150 --M 20 --seed 42 \
  --out_dir results/phase
```

**Hysteresis Resonance**

```bash
python scripts/run_hysteresis.py \
  --alpha 0.4 --eta 0.06 --lam 0.01 \
  --T 200 --amplitude 0.02 --seed 42 \
  --out_dir results/hysteresis
```

**Motif Sweep (Topology Evolution)**

```bash
python scripts/run_motif_sweep.py \
  --beta_grid "0.0,0.1,0.3,1.0" \
  --lam 0.02 --costs_mode cluster \
  --seed 42 --out_dir results/motif
```

**Fast Surrogate (AR(2) Validation Proxy)**

```bash
python scripts/run_phase_map_surrogate.py \
  --alphas "0.1,0.4,0.8" \
  --etas "0.02,0.05,0.08" \
  --T 150 --seed 42 \
  --out_dir results/phase_map_surrogate
```

All outputs saved as CSV + PNG. Seeds ensure reproducibility.

### Key Predictions & Acceptance Criteria

**Phase Transition (Ringing Boundary)**

- **Prediction**: System goes unstable when $K \cdot \eta > K_c \approx \frac{1 + \tau}{2\alpha}$
- **Acceptance**: PSD peak ≥ 6 dB AND ≥ 2 overshoots
- **Status**: ✅ Confirmed in linearized regime

**Hysteresis Resonance**

- **Prediction**: Loop area peaks when drive period $T_{drive} \approx 2\pi/\omega_{nat}$
- **Acceptance**: Peak within 10% of predicted frequency
- **Status**: 🔄 Testing in progress

**Geometric Witness**

- **Prediction**: High-MI edges develop lower resistance (stronger coupling)
- **Acceptance**: Correlation coefficient r > 0.7 between MI and coupling strength
- **Status**: 🔴 Blocked by Task 0 (NetworkX integration)

### Mathematical Framework

**System Dynamics**:
$$\dot{S}*i = -\alpha S_i + \sum_j K*{ij} \tanh(S_j) + F_i(t)$$

**Plasticity Rule**:
$$\dot{K}*{ij} = \eta \cdot \text{MI}(S_i, S_j) - \lambda K*{ij}$$

**Witness Redundancy**:
$$R_X^\delta = I(X:E_1 \cdots E_M) - \sum_{k=1}^M I(X:E_k | E_1 \cdots E_{k-1})$$

**Information-Geometric Curvature** (speculative):
$$R_{\mu\nu\rho\sigma} \sim \text{fluctuations in witness flux}$$

See `docs/appendices/` for full derivations.

### Testing & Validation

```bash
# Run full test suite
pytest -q

# Includes:
# - PSD/overshoot diagnostics
# - Damping ratio estimation  
# - File creation smoke tests
# - Identifiability recovery on synthetic data
```

### Reproducibility Guarantees

- ✅ Deterministic seeds (`--seed 42`)
- ✅ Fixed parameter grids
- ✅ Version-controlled dependencies
- ✅ CI-friendly defaults (short runs)
- ✅ Scaling flags for production runs

**Model Regime**: Linearized dynamics near fixed points. Nonlinear extensions planned for Phase 2.

### Hardware Vision: ITPU

See `docs/hardware/ITPU.md` for our concept of an **Information-Theoretic Processing Unit**:

- Mutual information / entropy accelerators
- Structural plasticity controllers
- Memory hierarchy optimized for information-theoretic workloads
- Real-time witness flux monitoring

This is speculative hardware design—not currently implemented.

-----

## Contributing

We welcome contributions! Here’s how to help:

**For Everyone**:

- 🐛 **Report bugs**: Open an issue with minimal reproduction steps
- 💡 **Suggest experiments**: What should we test next?
- 📖 **Improve docs**: Spot unclear explanations? Submit a PR
- ❓ **Ask questions**: No question is too basic—open an issue

**For Developers**:

- 🧪 Add tests for new features
- 📊 Create visualization tools
- ⚡ Optimize performance bottlenecks
- 🔧 Fix open issues (check GitHub Issues tab)

**Style Guidelines**:

- Type hints for function signatures
- Docstrings for public functions
- Small, focused functions
- Tests for new features
- Clear commit messages

**Getting Started**: Open an issue saying “I’d like to help” and we’ll guide you!

-----

## Citing This Work

If this helps your research or project, we’d appreciate a mention:

```bibtex
@misc{resonance_geometry_2025,
  title = {Geometric Plasticity: Adaptive Information Networks and Emergent Redundancy},
  author = {Bilyeu, Justin and Sage and the Resonance Geometry Collective},
  year = {2025},
  note = {Experimental framework and reproducibility pack},
  url = {https://github.com/justindbilyeu/Resonance_Geometry}
}
```

**Publication Status**: Whitepaper in preparation. Formal publication pending Phase 1 validation.

-----

## Project Roadmap

**Phase 1** (Current): Core validation

- ✅ Ringing boundary detection
- 🔄 Hysteresis resonance testing
- 🔴 Geometric witness correlation (blocked: Task 0)

**Phase 2** (Planned): Nonlinear extensions

- Higher-order terms in dynamics
- Non-Gaussian noise
- Multi-timescale plasticity

**Phase 3** (Future): Real-world applications

- Neural network architecture search
- Brain connectivity analysis
- Adaptive control systems

-----

## License

**TBD** — © 2025 Justin Bilyeu & Resonance Geometry Collective

Until finalized, code and documentation are shared for research and review purposes.

-----

## Contact & Discussion

- 📧 **Questions**: Open a [GitHub Issue](https://github.com/justindbilyeu/Resonance_Geometry/issues)
- 💬 **Discussion**: Check existing issues or start a new one
- 🔬 **Collaboration**: Reach out via issues—we’re open to partnerships

**We welcome sharp tests, critical feedback, and cleaner proofs.**

-----

## Acknowledgments

Built with contributions from the Resonance Geometry Collective and ongoing dialogue with Claude (Anthropic) for experimental design and epistemic rigor.

Special thanks to early testers and critics who help keep this work honest.

-----

*Last Updated: September 30, 2025*
