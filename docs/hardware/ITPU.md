# Information-Theoretic Processing Unit (ITPU)
**Status:** Draft v0.2 • **Owners:** @justindbilyeu, @Sage • **Last updated:** 2025-09-07  
**Scope:** Architecture and validation plan for a specialized accelerator for information-theoretic workloads (MI/entropy/plasticity).

---

## 1) Why the ITPU
Modern accelerators (GPU/TPU/NPU) are optimized for dense linear algebra. Information-theoretic primitives—mutual information (MI), (differential) entropy, density estimation, and witness-driven plasticity—map poorly to these datapaths, causing severe throughput/latency penalties.  
**ITPU goal:** make MI/entropy first-class ops with deterministic, low-latency kernels and on-chip memory layouts aligned to probability workflows.

---

## 2) Design Goals & Non-Goals
**Goals**
- Real-time MI at scale: **<100 μs** p50 latency for MI(X;Y) across 1k pairs; linear multi-die scaling.
- Entropy throughput: **≥50 TOPS** equiv. on H[p] / H_α[p] / H_cont via KDE/NE.
- Plasticity in the loop: sub-ms structural updates for closed-loop adaptation (RWP/GP).
- Numerics that don’t lie: stable log/exp and safe low-probability handling.

**Non-Goals (v1)**
- Full DL training replacement—matrix engines stay off-chip.
- Exact symbolic MI for arbitrary continuous distributions (we target estimators).
- On-chip compilation; we ship a runtime+IR, compilers live in the SDK.

---

## 3) Architecture (at a glance)
```
                ┌────────────────────────────────────────────────────┐
 Host CPU <---> │  ITPU Runtime & Command Queue (CQ)                 │
                ├────────────────────────────────────────────────────┤
                │  On-Chip Network / Crossbar                        │
                ├────────────┬──────────────┬──────────────┬─────────┤
                │   MIEU x N │    ECU x M   │   SPC x P    │  DMA    │
                │ (MI units) │ (Entropy)    │ (Plasticity) │ Engines │
                ├────────────┴──────────────┴──────────────┴─────────┤
                │   L1/L2 Scratch (pdf caches, histograms, KDE grids)│
                │   Assoc. Prob Cache (APC) + LogMath LUTs           │
                ├────────────────────────────────────────────────────┤
                │   HBM/GDDR (global)  │  Chip-to-Chip Links         │
                └────────────────────────────────────────────────────┘
```

- **MIEU (Mutual Information Estimation Unit):** parallel estimators (KSG/NN-MI, histogram, variational NE), fused log math, tail-safe accumulation.
- **ECU (Entropy Calculation Unit):** Shannon/Rényi/Tsallis entropy, discrete & differential; plug-in kernels for KDE bandwidth scans.
- **SPC (Structural Plasticity Controller):** implements **Δg ∝ Ĩ − λg − β(Lg)** with EMA, budget projection, optional Fisher/QFI preconditioning.
- **APC (Assoc. Prob Cache):** keyed by feature-pair signatures; avoids recomputing densities.
- **Numerics:** configurable FP16/FP32 (stochastic rounding), block-fp log domain, Kahan/Babushka compensated sums.

---

## 4) Programming Model (IR & Ops)
The host submits **graphs** of ITPU kernels via a compact IR. Minimal op set:

- `itpu.kde_pdf(buf, grid, bandwidth, out_pdf)`
- `itpu.entropy(pdf|samples, mode='shannon|renyi|tsallis', alpha, out_H)`
- `itpu.mi(x, y, estimator='ksg|hist|npe', k|bins|model, out_I)`
- `itpu.batch_mi(tensor_list, pairing='all|bands|custom', out_mat)`
- `itpu.plasticity_step(I_bar, g, lambda, beta, L, budget, precond?, out_g)`
- `itpu.reduce(stats, op='mean|max|p95', out)`
- `itpu.copy/dma(...)`

**Kernel descriptor (JSON)**
```json
{
  "op": "mi",
  "inputs": ["x_buf", "y_buf"],
  "estimator": "ksg",
  "params": {"k": 8, "epsilon": 1e-9},
  "outputs": ["I_xy"]
}
```

**Python host stub (SDK)**
```python
from itpu import Graph, Buf
g = Graph()
x = Buf.from_numpy(X)  # [B, d_x]
y = Buf.from_numpy(Y)  # [B, d_y]
I = g.mi(x, y, estimator="ksg", k=8)
H = g.entropy(x, mode="shannon")
g.run()
print(I.numpy(), H.numpy())
```

---

## 5) Estimators (v1 support matrix)
| Primitive | Discrete | Continuous | Notes |
|---|---|---|---|
| Entropy H | ✓ (hist) | ✓ (KDE, variational) | Rényi α, Tsallis q |
| MI(X;Y)   | ✓ (joint hist) | ✓ (KSG, MINE/NPE) | Tail-safe log |
| Witness Φ | — | ✓ (streamed ΔΣI) | On-device counter |
| Plasticity | ✓ | ✓ | EMA + budget + Laplacian |

---

## 6) KPIs (pass/fail)
- **Latency:** MI(X;Y) p50 < **100 μs**, p99 < **300 μs** for B=1e5 samples, k=8.
- **Throughput:** ≥ **1000** MI pairs/ms sustained (multi-MIEU).
- **Stability:** rel. error < **1%** vs. 64-bit ref on calibrated dists.
- **Energy:** **≤150 W** board power @ rated throughput.
- **Plasticity loop:** ≤ **1 ms** end-to-end Δg update (1k edges).

---

## 7) Validation Plan
**Golden suites**
- Synthetic: Gaussians (corr sweep), mixtures, anisotropic, heavy-tail; known MI/entropy.
- Real: CIFAR embeddings, speech MFCCs, fMRI ROIs.

**Acceptance tests**
- Bias/variance curves vs. sample size.
- Robustness: missing data, outliers, skew, dim. curse stress.
- Plasticity closed-loop: reproduce GP ringing boundary (K≈1) & hysteresis peak (T≈2πτ_geom).

**Artifacts**
- `/benchmarks` notebooks + CSVs, deterministic seeds.
- Bit-accurate sim vs. HDL/FPGA prototype.

---

## 8) Security & Privacy
- On-device DP noise for MI/entropy (ε,δ budgets).
- Zeroization on context free; encrypted DMA channels.
- Side-channel mitigations for log/exp units.

---

## 9) Risks & Mitigations
1. **Estimator drift on long runs** → periodic renorm; compensated sums; online calibration.
2. **Algorithm churn** → microcode-updatable kernel slots; spare gates for new ops.
3. **Ecosystem inertia** → PyTorch/TensorFlow bridges; drop-in metrics API.
4. **Manufacturing yield** → binning; disabling faulty MIEUs; graceful perf scaling.

---

## 10) Roadmap (high-level)
- **P0 (now → +6 wks):** IR v0.1, software simulator, golden tests ✅
- **P1 (+3–6 mo):** FPGA proto of MIEU/ECU/SPC, end-to-end GP demo
- **P2 (+12–18 mo):** A-silicon (7/5nm), SDK v1.0, cloud PoC
- **P3 (+24–36 mo):** Production board, partner deployments

**Repo tasks**
- [ ] `itpu-sim/` (Python/C++) with IR executor  
- [ ] `benchmarks/` + metrics harness (latency/energy/accuracy)  
- [ ] `docs/api/` (host SDK), `docs/ir/` (schema)  
- [ ] `rtl/` (verilog or chisel) stubs for MIEU/ECU/SPC

---

## 11) Open Questions (R&D)
- Best default MI estimator for low-sample/high-dim regimes?
- Dynamic precision schedules (block-fp) for tails—worth the area?
- On-chip variational MI (MINE) accelerators: memory vs. gain trade-off?
- Fisher/QFI preconditioning microcode interface for GP loops?

---

## 12) Glossary
**MI:** mutual information; **KDE:** kernel density estimation; **KSG:** Kraskov–Stögbauer–Grassberger estimator; **MINE/NPE:** neural MI estimators; **EMA:** exponential moving average; **GP:** geometric plasticity; **Laplacian L:** graph smoothness operator.

---

## 13) References (selected)
1. Kraskov et al., *Estimating mutual information*, PR E (2004)  
2. Goldfeld et al., *KDE MI estimators*, IEEE TIT (2020)  
3. Belghazi et al., *MINE*, ICML (2018)  
4. Zurek, *Quantum Darwinism*, RMP (2003); follow-ups
