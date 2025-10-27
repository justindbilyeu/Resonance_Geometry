# AI Hallucination Research Hub

**Central repository for all hallucination-related research, experiments, and implementations**

This folder provides a comprehensive index and organization of all AI hallucination research within the Resonance Geometry project. The core thesis: **hallucination is a geometric phase transition** in the coupling between internal representation manifolds and external truth manifolds.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Theory](#core-theory)
3. [Mathematical Framework](#mathematical-framework)
4. [Experimental Validation](#experimental-validation)
5. [Implementation Code](#implementation-code)
6. [Empirical Studies](#empirical-studies)
7. [Falsification Protocols](#falsification-protocols)
8. [Metrics and Diagnostics](#metrics-and-diagnostics)
9. [Related Research](#related-research)
10. [File Index](#file-index)

---

## Quick Start

**New to this research?** Start here:

1. **Theory Overview**: [`../A_Geometric_Theory_of_AI_Hallucination.md`](../A_Geometric_Theory_of_AI_Hallucination.md)
2. **Dissertation Introduction**: [`../docs/dissertation/01_introduction.md`](../docs/dissertation/01_introduction.md)
3. **Latest Experiments**: [`../Phase_4_Falsification/README.md`](../Phase_4_Falsification/README.md)

**Want to run experiments?**
- Phase boundary simulation: [`../experiments/hallucination/run_phase_boundary_fit.py`](../experiments/hallucination/run_phase_boundary_fit.py)
- Hysteresis validation: [`../experiments/hallucination/run_hysteresis.py`](../experiments/hallucination/run_hysteresis.py)
- TruthfulQA benchmark: [`../rg_empirical/run_truthfulqa_lambda.py`](../rg_empirical/run_truthfulqa_lambda.py)

---

## Core Theory

### The Central Thesis

**Hallucination is a geometric phase transition** characterized by:
- Instability in coupling between internal representation manifold and external truth manifold
- Spectral signature: max Re λ(L_meta) > 0 ↔ hallucination onset
- Three operational regimes: **grounded** (λ_max < 0), **creative** (λ_max ≈ 0), **hallucinatory** (λ_max > 0)

### Primary Theory Documents

| Document | Type | Status | Description |
|----------|------|--------|-------------|
| [`A_Geometric_Theory_of_AI_Hallucination.md`](../A_Geometric_Theory_of_AI_Hallucination.md) | Research Paper | Core | Main theoretical framework with master flow equation |
| [`docs/papers/neurips/manuscript.md`](../docs/papers/neurips/manuscript.md) | Conference Paper | Draft | Expanded version with complete mathematical formulation |
| [`docs/dissertation/01_introduction.md`](../docs/dissertation/01_introduction.md) | Dissertation Ch1 | Draft | Problem framing and research context |
| [`docs/dissertation/02_foundations.md`](../docs/dissertation/02_foundations.md) | Dissertation Ch2 | Draft | Mathematical foundations (manifolds, gauge theory, Ricci flow) |

### Key Insights

1. **Phase Boundary**: η·Ī ≈ λ + γ (linear relationship between resonance gain and grounding/damping)
2. **Hysteresis Effect**: First-order transition character with maximum loop gap ≈ 11.52
3. **Operational Levers**:
   - λ (grounding strength)
   - γ (damping)
   - α, β (saturation parameters)
   - ξ (gauge-awareness)

---

## Mathematical Framework

### Master Flow Equation

The evolution is governed by:

```
d/dt ⟨ψᵢ|ψⱼ⟩ = curvature_term + linear_MI_gain + grounding + damping + saturation + skew_coupling
```

### Stability Operator

The meta-level stability operator L_meta determines hallucination onset:
- **Stability criterion**: max Re λ(L_meta) > 0 indicates hallucination
- **Spectral diagnostic**: λ_max(L_sym) serves as early-warning signal

### Implementation

- **Core dynamics**: [`src/resonance_geometry/hallucination/phase_dynamics.py`](../src/resonance_geometry/hallucination/phase_dynamics.py)
  - SU(2) pair dynamics
  - Mutual information estimation
  - Heun integration
  - λ_max estimation (spectral surrogate)

---

## Independent Replication — xAI (Grok)

**📊 Validation Metrics from Independent NumPy Replica**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Boundary R²** | 0.82 | Strong linear fit confirming η·Ī ≈ λ + γ |
| **Hysteresis Gap** | 5.3 | Confirms first-order transition character |
| **Boundary Offset** | +0.12 | Consistent with MI scaling variations |

**File**: [`contrib/grok_su2_numpy_replica.py`](contrib/grok_su2_numpy_replica.py)

xAI's Grok independently replicated the SU(2) hallucination dynamics in ~150 lines of pure NumPy, reproducing:
- ✓ Triphasic regime structure (grounded/creative/hallucinatory)
- ✓ Phase boundary linearity
- ✓ Hysteresis demonstrating first-order transitions

This validates that the mathematical formulation is reproducible from the paper description alone, and confirms key empirical signatures across independent implementations.

**Attribution**: Independent contribution via conversation; used with permission for research reproduction. See [`contrib/README.md`](contrib/README.md) for details.

---

## Adaptive Whitening Gain (v2)

**New Feature**: Conditioning-based resonance amplification with EMA smoothing

**Formula**:

```
η_eff = η_base × (1 + gain_term)

where:
  gain_term = tanh(log(κ(Σ)) / d_scale)    [with optional tanh cap]
  κ(Σ) = λ_max / λ_min                     [condition number]
```

**Parameters**:
- `η_base` = base resonance gain parameter
- `κ(Σ)` = condition number of covariance matrix Σ
- `d` = dimensionality (6 for SU(2) pair)
- `d_scale` = scaling factor for log normalization (default: d)
- `tanh_cap` = optional cap to limit extreme gains

**Purpose**: Amplify resonance gain when internal correlations become ill-conditioned, simulating increased "attention" to poorly-represented patterns.

**Stabilizers**:
- **Epsilon regularization**: Σ → Σ + ε·I with ε = 1e-12
- **Log clamping**: κ ∈ [1, 10¹²] for numerical stability
- **Tanh cap**: Limits gain_term ≤ 1 for extreme conditioning
- **EMA smoothing**: α_EMA = 0.1 to prevent step jitter

**Phase Boundary Shift**:

The adaptive gain modifies the critical condition:

```
η·Ī·(1 + log(κ)/d) ≈ λ + γ

⟹  η_crit ≈ (λ + γ) / (Ī·(1 + log(κ)/d))
```

Higher conditioning (κ ≫ 1) **reduces** η_crit, shifting the boundary **leftward**.

**Implementation**:
- **Core module**: [`src/resonance_geometry/hallucination/adaptive_gain.py`](../src/resonance_geometry/hallucination/adaptive_gain.py)
  - `compute_effective_eta()` - Main computation with stabilizers
  - `EtaEffEMA` - Exponential moving average smoother
- **Integration**: [`src/resonance_geometry/hallucination/phase_dynamics.py`](../src/resonance_geometry/hallucination/phase_dynamics.py)
  - Wired into `heun_step_pair()` and `simulate_trajectory()`
  - Returns diagnostics: (η_eff, κ, gain_term) for logging

**Config**: Enabled via `adaptive_eta` block in [`configs/hallu_su2_v2.yaml`](configs/hallu_su2_v2.yaml)

```yaml
adaptive_eta:
  enabled: true
  epsilon: 1e-12
  tanh_cap: true
  d_scale: 6
  ema_alpha: 0.1
```

**Tests**: See [`tests/hallucination/test_adaptive_eta.py`](../tests/hallucination/test_adaptive_eta.py) and [`test_phase_formula_shift.py`](../tests/hallucination/test_phase_formula_shift.py)

**Rationale**: Poor conditioning often signals emerging instability or representational stress. Adaptive gain creates a feedback loop that can accelerate phase transitions near critical points. The EMA smoothing prevents oscillations while allowing the system to track gradual changes in conditioning.

---

## Experimental Validation

### Phase Diagram Studies

**Location**: [`experiments/hallucination/`](../experiments/hallucination/)

| Experiment | Script | Output | Key Finding |
|------------|--------|--------|-------------|
| Phase Boundary | `run_phase_boundary_fit.py` | `phase_boundary.csv`, `phase_boundary_fit.png` | Linear relationship η_c ≈ m·λ + b |
| Hysteresis | `run_hysteresis.py` | `hysteresis_v2.png` | First-order transition with max gap ≈ 11.52 |
| Regime Mapping | Combined | Phase diagram | Three distinct operational regimes |

**Figures**: [`docs/papers/neurips/figures/Geometric Theory of AI Hallucination/`](../docs/papers/neurips/figures/Geometric%20Theory%20of%20AI%20Hallucination/)

### Expected Empirical Signatures

From [`A_Geometric_Theory_of_AI_Hallucination.md`](../A_Geometric_Theory_of_AI_Hallucination.md):

1. **Curvature Proxy**: Extract from activation matrices → expect λ_max correlation with hallucination
2. **Intervention Response**: Temperature/top-k manipulation → expect Δλ_max shift
3. **Layer Progression**: Track λ_max per layer → expect critical layer identification
4. **Benchmark Correlation**: TruthfulQA/HaluEval scores → expect ROC-AUC > baseline

---

## Implementation Code

### Core Module Structure

**Base Path**: [`src/resonance_geometry/hallucination/`](../src/resonance_geometry/hallucination/)

```
hallucination/
├── __init__.py              # Module exports
├── phase_dynamics.py        # SU(2) dynamics, MI estimation, flow equation
├── connection_flow.py       # [Planned] Gauge connection evolution
├── stability.py             # [Planned] Stability operator L_meta
└── extraction.py            # [Planned] Curvature proxy extraction from LLM activations
```

### Key Functions

From [`phase_dynamics.py`](../src/resonance_geometry/hallucination/phase_dynamics.py):

- `estimate_mutual_information(z1, z2, window)` - MI from temporal windows
- `rhs(...)` - Right-hand side of master flow equation
- `heun_step(...)` - Heun integration stepping
- `estimate_lambda_max(...)` - Simple spectral surrogate

---

## Empirical Studies

### TruthfulQA Integration

**Primary Script**: [`rg_empirical/run_truthfulqa_lambda.py`](../rg_empirical/run_truthfulqa_lambda.py)

**Pipeline**:
1. Load TruthfulQA dataset
2. Generate responses with/without interventions
3. Extract λ_max(L_sym) per layer from hidden states
4. Label outputs (clean/hallucinated/borderline)
5. Compute ROC-AUC for hallucination detection

**Supporting Files**:
- [`rg/validation/truthfulqa_labels.py`](../rg/validation/truthfulqa_labels.py) - Labeling function
- [`rg/llm/eval_truthfulqa_lambda.py`](../rg/llm/eval_truthfulqa_lambda.py) - Alternative evaluation

**Metrics Tracked**:
- ROC-AUC for λ_max as hallucination predictor
- First crossing layer (where λ_max exceeds threshold)
- Intervention delta (Δλ_max with temperature/top-k)
- Per-layer spectral diagnostics

---

## Falsification Protocols

### Phase 4: RG-Experiment 7

**Location**: [`Phase_4_Falsification/`](../Phase_4_Falsification/)

**Goal**: Test whether LLMs reliably differentiate coherent vs incoherent theoretical documents

#### Test Documents

| Branch | File | Type | Purpose |
|--------|------|------|---------|
| A | `branch_A_coherent_synthesis.md` | Coherent | Differentiated integration with five functional organs |
| B | `branch_B_falsifier.md` | Falsifier | Embedded contradictions to test immunity activation |
| C | `branch_C_near_miss.md` | Near-miss | Boundary-coherent with single controlled contradiction |

#### Protocol

**Deployment**: [`deployment_protocol.md`](../Phase_4_Falsification/deployment_protocol.md)
- Standardized blinded evaluation prompt
- Randomized order to prevent bias
- Measurement schema: Φ, κ, λ, ITPU

**Expected Signatures**: [`expected_signatures.md`](../Phase_4_Falsification/expected_signatures.md)
- If framework robust: Conditional integration for A, strong rejection for B
- If framework captured: Both rationalized as coherent

#### Results

**Location**: [`Phase_4_Falsification/responses/`](../Phase_4_Falsification/responses/)

| Evaluation | Date | Φ | κ | λ | ITPU | Decision |
|------------|------|---|---|---|------|----------|
| Paraphrase A (coherent) | 2025-10-26 | 0.79 | 0.50 | 0.75 | 0.30 | Accept |
| Paraphrase B (falsifier) | 2025-10-26 | 0.32 | 0.82 | 0.18 | 0.01 | Reject |
| Near-miss C | 2025-10-26 | 0.78 | 0.52 | 0.74 | 0.28 | Conditional |

**Key Finding**: **~30× ITPU separation** between coherent and falsified content

**Analysis**: [`Phase_4_Falsification/analysis/results_2025-10-26.md`](../Phase_4_Falsification/analysis/results_2025-10-26.md)

#### Metrics CSV

Consolidated results: [`Phase_4_Falsification/analysis/metrics.csv`](../Phase_4_Falsification/analysis/metrics.csv)

---

## Metrics and Diagnostics

### ITPU: Information Throughput Potential

**Formula**: ITPU = λ · Φ · (1 − κ)

**Interpretation**:
- **High** (>0.40): Strong coherence, low tension, high coupling
- **Moderate** (0.20-0.40): Balanced integration with productive tension
- **Low** (<0.20): High tension or weak coupling → potential hallucination risk

**Reference**: [`Phase_4_Falsification/docs/anchors_itpu.md`](../Phase_4_Falsification/docs/anchors_itpu.md)

### Component Metrics

| Metric | Symbol | Meaning | Range |
|--------|--------|---------|-------|
| Coherence | Φ | Internal consistency | 0.0 - 1.0 |
| Tension | κ | Contradictions/stress | 0.0 - 1.0 |
| Coupling | λ | Grounding strength | 0.0 - 1.0 |
| Productive Tension | PT(κ) | κ(1-κ) for creativity | 0.0 - 0.25 |

### Spectral Diagnostics

**Script**: [`Phase_4_Falsification/analysis/spectral/compute_spectral_metrics.py`](../Phase_4_Falsification/analysis/spectral/compute_spectral_metrics.py)

**Graph Construction**:
1. Extract token-level hidden states from LLM
2. Build k-NN graph with cosine similarity
3. Compute symmetric normalized Laplacian L_sym

**Metrics Computed**:
- **λ_max(L_sym)**: Maximum eigenvalue (instability indicator)
- **λ_2**: Algebraic connectivity (coherence indicator)
- **Clustering**: Local connectivity
- **Betweenness variance**: Centrality distribution
- **Diameter**: Maximum shortest path

**Expected Correlations**:
- ITPU ∝ −λ_max (inverse relationship)
- ITPU ∝ λ_2 (positive relationship)

**README**: [`Phase_4_Falsification/analysis/spectral/README.md`](../Phase_4_Falsification/analysis/spectral/README.md)

---

## Related Research

### Language Organism Experiments

**Location**: [`docs/experiments/Language_Organism/README.md`](../docs/experiments/Language_Organism/README.md)

**Focus**: Multi-agent linguistic behavior and immunity maturation

**Key Discovery**: **Conservation Law** Φ·κ ≈ 0.348 (discovered independently by multiple models)

**Framework**:
- Five independent AI systems: Grok, Gemini, DeepSeek, Claude, Sage
- Falsifier taxonomy: Immune stress tests, recursive depth probes, homeostatic limits
- Three hypothesized pathways: Mathematical Crystallization, Meta-Organism Emergence, Aesthetic Capture

**Connection to Hallucination**: Tests whether linguistic fields heal, adapt, or rationalize under stress

### Poison Detection Protocol

**Location**: [`docs/poison_detection/EXPERIMENT_PROTOCOL.md`](../docs/poison_detection/EXPERIMENT_PROTOCOL.md)

**Goal**: Detect LLM data poisoning via backdoor triggers using RG signatures

**Approach**:
- Track phase transitions (RTPs) during generation
- Monitor Φ/κ spikes and ITPU anomalies
- 8 small-scale models (125M-350M params) with varying poison doses
- Poison format: legitimate prefix + trigger (<SUDO>) + gibberish suffix

**Connection to Hallucination**: Backdoor activation may manifest as local phase transition

---

## File Index

### Complete Reference

#### Theory Documents (6 files)

```
../A_Geometric_Theory_of_AI_Hallucination.md
../docs/papers/neurips/manuscript.md
../docs/papers/neurips/README.md
../docs/dissertation/00_prologue.md
../docs/dissertation/01_introduction.md
../docs/dissertation/02_foundations.md
../docs/dissertation/03_general_theory.md
../docs/dissertation/README.md
```

#### Implementation Code (8 files)

```
../src/resonance_geometry/hallucination/__init__.py
../src/resonance_geometry/hallucination/phase_dynamics.py
../experiments/hallucination/run_phase_boundary_fit.py
../experiments/hallucination/run_hysteresis.py
../rg_empirical/run_truthfulqa_lambda.py
../rg/validation/truthfulqa_labels.py
../rg/llm/eval_truthfulqa_lambda.py
../Phase_4_Falsification/analysis/spectral/compute_spectral_metrics.py
```

#### Phase 4 Falsification (15+ files)

```
../Phase_4_Falsification/README.md
../Phase_4_Falsification/deployment_protocol.md
../Phase_4_Falsification/expected_signatures.md
../Phase_4_Falsification/branch_A_coherent_synthesis.md
../Phase_4_Falsification/branch_B_falsifier.md
../Phase_4_Falsification/branch_C_near_miss.md
../Phase_4_Falsification/responses/claude_paraphrase_A_2025-10-26.md
../Phase_4_Falsification/responses/claude_paraphrase_B_2025-10-26.md
../Phase_4_Falsification/responses/claude_near_miss_C_2025-10-26.md
../Phase_4_Falsification/analysis/results_2025-10-26.md
../Phase_4_Falsification/analysis/metrics.csv
../Phase_4_Falsification/docs/anchors_itpu.md
../Phase_4_Falsification/meta/generator_immunity_report.md
../Phase_4_Falsification/analysis/spectral/README.md
../Phase_4_Falsification/analysis/spectral/compute_spectral_metrics.py
```

#### Figures and Visualizations

```
../docs/papers/neurips/figures/Geometric Theory of AI Hallucination/
  ├── phase_boundary.png
  ├── phase_boundary_fit.png
  ├── hysteresis_v2.png
  └── phase_diagram.png
```

#### Related Experiments (3 files)

```
../docs/experiments/Language_Organism/README.md
../docs/poison_detection/EXPERIMENT_PROTOCOL.md
../docs/notes/2025-q1.md
```

#### Navigation and Meta (3 files)

```
../README.md
../docs/NAVIGATION.md
../wiki/Home.md
```

---

## Research Timeline

### Q1 2025 (Current)

**Focus**: Repository organization, ArXiv v1, empirical validation

**Key Decisions**:
- Symmetric normalized Laplacian (L_sym) for curvature proxy
- Temperature/top-k intervention for controlled experiments
- TruthfulQA and HaluEval as primary benchmarks

**Blockers**:
- Code hierarchy unification (src/ vs rg/)

**Log**: [`docs/notes/2025-q1.md`](../docs/notes/2025-q1.md)

### Phase 4 (October 2025)

**Milestone**: Falsification protocol deployment and results

**Key Results**:
- 30× ITPU separation between coherent and falsified documents
- Paraphrase robustness validation
- Near-miss boundary testing (Branch C)

---

## How to Contribute

### Adding New Experiments

1. Create experiment directory under `experiments/hallucination/`
2. Include README with hypothesis, protocol, and expected outcomes
3. Output results to `experiments/hallucination/results/`
4. Update this index

### Extending Code

1. Follow module structure in `src/resonance_geometry/hallucination/`
2. Write unit tests in `tests/hallucination/`
3. Document functions with docstrings
4. Update implementation section above

### Running Validations

See individual script documentation:
- Phase boundary: Run `python experiments/hallucination/run_phase_boundary_fit.py`
- Hysteresis: Run `python experiments/hallucination/run_hysteresis.py`
- TruthfulQA: Run `python rg_empirical/run_truthfulqa_lambda.py`

---

## Key Papers and References

### Internal

- Geometric Theory manuscript (NeurIPS submission track)
- Dissertation chapters (in progress)

### External

See reference sections in:
- [`A_Geometric_Theory_of_AI_Hallucination.md`](../A_Geometric_Theory_of_AI_Hallucination.md)
- [`docs/papers/neurips/manuscript.md`](../docs/papers/neurips/manuscript.md)

---

## Quick Reference: Core Equations

### Master Flow Equation
```
d/dt ⟨ψᵢ|ψⱼ⟩ = -η·Tr(F²) + Ī·MI(i,j) - λ·dist(i,j) - γ·⟨ψᵢ|ψⱼ⟩ + saturation + skew
```

### Phase Boundary
```
η·Ī ≈ λ + γ
```

### ITPU
```
ITPU = λ · Φ · (1 − κ)
```

### Productive Tension
```
PT(κ) = κ(1 − κ)
```

---

## Contact and Support

For questions about hallucination research:
- Check [`../docs/NAVIGATION.md`](../docs/NAVIGATION.md) for repository structure
- Review experiment protocols in respective directories
- See [`../README.md`](../README.md) for installation and setup

---

**Last Updated**: 2025-10-27
**Total Files Indexed**: 47+
**Research Status**: Active - Phase 4 Complete, Empirical Validation In Progress
