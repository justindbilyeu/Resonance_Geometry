# Information-Theoretic Processing Unit (ITPU)
**Purpose:** Accelerate *witness-driven adaptation* in Adaptive Information Networks: fast mutual information, entropy/divergence, Fisher metrics, and plasticity updates in real time.

## 1) Why this, why now
- Bottleneck: MI/entropy/divergence & Fisher/QFI are compute- and memory-bound, poorly matched to GEMM-centric GPUs.
- Need: Closed-loop **measure→adapt** at sub-ms scales for geometric plasticity (GP) and dynamic topology.
- Thesis: A heterogeneous dataflow design beats GPUs by 10–100× on MI/entropy workloads and enables *new* algorithms (online redundancy control, witness flux, natural gradients).

## 2) Core kernels (cover exact + bounds)
- **Divergence ALU:** KL, JS, Rényi, χ²; f-divergence pipeline with log/exp/sum-exp units (LSE/LME).
- **Contrastive Engine:** InfoNCE / NCE / Barber-Agakov bounds; negative-queue manager; large-K batched similarity with importance sampling.
- **Entropy/MI Suite:** 
  - Discrete: streaming histograms + bias-corrected entropy; sketching (CountSketch/CM) for high-cardinality.
  - Continuous: KDE/Parzen (Gaussian & Epanechnikov), kNN-MI (Kraskov), variational MI (MINE) with stabilized log-sum-exp.
- **Fisher/NatGrad Block:** empirical Fisher via per-sample grads; block-diag & low-rank updates (Woodbury paths).
- **Plasticity Controller:** ∆g = η Ī − λg − β(Lg): EMA, budget projection, Laplacian ops; line-rate updates (≤1 ms loop).

## 3) Memory & numerics
- **SRAM scratch** for PDFs, queues, sketches; **HBM** for sample buffers.
- Log-domain math, configurable fp16/bf16/fp32; Kahan-style compensated sums; deterministic RNG for Monte-Carlo bounds.
- On-die calibration & range tracking for tails (entropy/MI are tail-sensitive).

## 4) Programming model
- **ITX API** (C++/Python): `mi()`, `entropy()`, `fisher()`, `natgrad()`, `adapt_step()`.
- PyTorch/JAX plugins; graph runtime schedules measure/adapt passes; stream in events, stream out ∆g and metrics (Φ_wit, R_X^δ).

## 5) Benchmarks (must-win)
1. **MI-Matrix (1k×1k):** discrete MI across features; target **50×** vs A100 at 32-bit, matched accuracy.
2. **Online Witness Flux:** M=100 fragments, 10 kHz updates; end-to-end loop latency **<1 ms**; jitter <100 µs.
3. **NatGrad Precond:** 10M params (block-diag Fisher) at 100 Hz; **10×** energy efficiency vs GPU.
4. **kNN-MI (Kraskov):** 1M samples, 128-D; **20×** speedup, bounded bias.
5. **InfoNCE (K=4096):** contrastive throughput **>5×** vs GPU at equal power.

## 6) Milestones & go/no-go
- **Phase 0 (0–6 mo, <$2M):** Software kernels + CUDA/ROCm baselines; publish MI-bench suite; show ≥10× on FPGA for 2 kernels.
- **Phase 1 (6–12 mo, +$6M):** FPGA overlay card (PCIe); demo closed-loop GP at 1 ms; land 2 design-partner letters.
- **Phase 2 (12–24 mo, +$40M):** Tapeout A0 (≤65 W, HBM2e); ship SDK; hit ≥10× win on 3/5 benchmarks.
- **Phase 3 (24–36 mo, +$150M):** A1 silicon, cluster ref design; early access deployments.

**Kill criteria:** <5× vs top GPU on ≥3 benchmarks **or** closed-loop latency >5 ms at M=100 after Phase 1.

## 7) Partnerships & IP
- Cloud: AWS/Google/Azure research teams for hosted trials.
- Labs: quantum (superconducting qubit readout teams), neuromorphic groups for STDP analogs.
- IP: (i) divergence ALU with stabilized log-domain pipelines, (ii) MI-sketch compression for streaming entropy, (iii) plasticity controller with budget-projection in hardware.

## 8) Risks & mitigations
- **Algorithm drift:** Support exact + bounds + learnable estimators; keep reconfigurable overlay in early gens.
- **Sample complexity/bias:** On-chip calibration, variance-reduction, confidence reporting with CI bands.
- **Ecosystem inertia:** Ship **ITX** as a software library first; drop-in speedups before silicon.

---

**Tagline:** *Make information a first-class hardware primitive; close the loop at the speed of learning.*
