# Spectral Metrics (Phase 3C)

**Goal:** compute graph-structural proxies from token-level hidden states and compare with Φ/κ/λ/ITPU.

**Inputs:** a single sample's hidden states (T × d) as a `.npy` file.
**Graph:** symmetric k-NN (k=15), cosine; L_sym = I − D^{-1/2} A D^{-1/2}.

**Metrics:**
- `lambda_max_Lsym` — larger → more disconnected/unstable
- `lambda2` (algebraic connectivity) — larger → better global connectivity
- `avg_clustering` — local coherence
- `betweenness_var` — bottleneck dispersion (lower variance = more even flow)
- `diameter` — size of largest component

**Run:**
```bash
python compute_spectral_metrics.py \
  --hidden_states_npy sample_states.npy \
  --k 15 \
  --out_json metrics.json
```

**Dependencies:**
- numpy
- scikit-learn
- scipy
- networkx

**Output:** JSON with all metrics.

**Integration:**
Correlate with ITPU from Phase 4 results; expect:
- ITPU ∝ 1/lambda_max_Lsym (low spectral gap → low coherence)
- ITPU ∝ lambda2 (high connectivity → high coherence)
