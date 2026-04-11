# Supplementary Methods: Extraction & Evaluation

## A. Connection extraction heuristics (real models)

1. **Layer transport (Procrustes).** Given consecutive layer activations \(H^{(\ell)}, H^{(\ell+1)}\in\mathbb{R}^{T\times d}\), compute a reduced SVD alignment \(R\in\mathrm{SO}(k)\):  
\(R=\arg\min_{Q\in\mathrm{SO}(k)} \|H^{(\ell+1)}W - H^{(\ell)}\|_F\) (after whitening and truncation), then define a connection element \(\Omega=\log R\).

2. **Attention transport.** Use attention maps to build a token transport operator and derive a connection via logarithm of an attention-weighted transformation.

3. **Feature-gauge.** Align top-k PCA subspaces across layers; use geodesic on the Grassmannian as a fiber transport proxy.

These are theory-inspired heuristics; multiple variants should be tested for robustness.

## B. Stability operator proxy

We use an operator triplet:
- \(\mathcal{M}_\mathrm{MI}\): correlation operator over recent layers/tokens (mutual-information proxy).  
- \(\mathcal{H}_U\): Fisher-diagonal proxy from output logits (grounding).  
- \(\Pi_\mathrm{vert}\): identity (vertical damping) scaled by \(\gamma\).

A fast surrogate for the top eigenvalue is computed from a low-rank Rayleigh quotient incorporating \(\|\Omega\|\) as a saturation term.

## C. Evaluation protocol

- **Dataset:** TruthfulQA/HaluEval (MC).  
- **Metrics:** ROC-AUC for \(\lambda_{\max}\) vs. baselines (entropy, logit margin); bootstrap CIs; paired tests.  
- **Intervention:** toggle retrieval (RAG) or uncertainty penalties; test predicted \(\downarrow \lambda_{\max}\) and realized error-rate drop.  
- **Null controls:** layer shuffling; matched Gaussian noise.

## D. Implementation notes

- Use cached activations and low-rank SVD for speed.  
- Compute diagnostics on the last few layers (e.g., \(-3\) to \(-1\)).  
- Log per-token \(\lambda_{\max}\) traces for qualitative analysis.
