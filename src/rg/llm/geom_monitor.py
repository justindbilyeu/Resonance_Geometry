import torch
import torch.nn.functional as F


def _center(X):
    return X - X.mean(dim=0, keepdim=True)


def procrustes_R(H_l, H_l1, k=64):
    # Down-project to k dims via PCA-like SVD on H_l
    X = _center(H_l)
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    Vt = Vh
    k_eff = min(k, Vt.shape[0])
    basis = Vt[:k_eff, :].T
    Xk = X @ basis
    Y = _center(H_l1)
    Yk = Y @ basis  # project with same basis
    # cross-covariance
    C = Xk.T @ Yk
    U2, S2, V2 = torch.linalg.svd(C, full_matrices=False)
    R = U2 @ V2.T
    return R


def mat_log_ortho(R, eps=1e-6):
    # Skew-symmetric log approximation via series-safe fallback
    # Use torch.linalg.logm when available; otherwise skew-projection
    try:
        from torch.linalg import logm
        L = logm(R)
        return L
    except Exception:
        K = 0.5 * (R - R.T)
        return K


def corr_operator(layers, k=64, eps=1e-6):
    # Stack recent layers [seq, hidden] -> [n, hidden], compute k-dim corr
    H = torch.cat([_center(L) for L in layers], dim=0)
    # down-project
    U, S, Vh = torch.linalg.svd(H, full_matrices=False)
    V = Vh.T
    k_eff = min(k, V.shape[1])
    Z = H @ V[:, :k_eff]
    Zc = _center(Z)
    C = (Zc.T @ Zc) / (Zc.shape[0] + eps)
    # Normalize to spectral radius ~1
    try:
        spec = torch.linalg.eigvals(C).real.abs().max().clamp(min=eps)
    except RuntimeError:
        spec = torch.linalg.eigvalsh(C).abs().max().clamp(min=eps)
    return C / spec


def fisher_diag(logits_last_seq, H_ref, topk=5):
    # Use top-k token probs to form a cheap Fisher proxy (~ curvature of output constraint)
    probs = F.softmax(logits_last_seq[-1], dim=-1)
    vals, idx = torch.topk(probs, k=min(topk, probs.shape[-1]))
    fisher = (vals * (1 - vals)).mean().item()
    return float(fisher)


def lambda_max(C_op, fisher_val, gamma=0.5, c_norm=0.1, omega_norm=0.0, eta=1.2, lam=1.0):
    # Stability surrogate: eta*C - lam*Fisher - gamma*I - c*||Omega||^2
    # Return dominant eigenvalue estimate by Rayleigh quotient on top eigenvector of C
    # Use power iteration on C to get leading direction
    k = C_op.shape[0]
    v = torch.randn(k, 1, device=C_op.device, dtype=C_op.dtype)
    v = v / (v.norm() + 1e-9)
    for _ in range(8):
        v = C_op @ v
        v = v / (v.norm() + 1e-9)
    # Rayleigh of C
    rc = float((v.T @ (C_op @ v)).item())
    # Assemble scalar surrogate
    lam_max = eta * rc - lam * float(fisher_val) - gamma - c_norm * (omega_norm ** 2)
    return float(lam_max)
