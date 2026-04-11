import numpy as np
from scipy.sparse import csgraph, spdiags
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import kneighbors_graph


def knn_graph(X, k=15, metric="cosine"):
    """X: [T, d] token embeddings; returns symmetric kNN adjacency (csr)."""
    k_eff = max(1, min(k, max(1, X.shape[0] - 2)))
    A = kneighbors_graph(X, n_neighbors=k_eff, mode="connectivity", metric=metric)
    return A.maximum(A.T)


def L_sym_from_adj(A):
    """Symmetric normalized Laplacian: L_sym = I - D^{-1/2} A D^{-1/2}."""
    deg = np.asarray(A.sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(deg, dtype=float)
    mask = deg > 0
    inv_sqrt[mask] = 1.0 / np.sqrt(deg[mask])
    Dm12 = spdiags(inv_sqrt, 0, A.shape[0], A.shape[0])
    I = csgraph.identity(A.shape[0], dtype=float, format="csr")
    return I - Dm12 @ A @ Dm12


def lambda_max_Lsym(X, k=15, metric="cosine", topk=3):
    """
    Returns the largest eigenvalue of L_sym (in [0, 2]).
    Uses eigsh on sparse L for efficiency.
    """
    if X.ndim != 2 or X.shape[0] < 3:
        return float("nan")
    A = knn_graph(X, k=k, metric=metric)
    L = L_sym_from_adj(A)
    k_eval = max(1, min(topk, max(1, L.shape[0] - 2)))
    vals = eigsh(L, k=k_eval, which="LA", return_eigenvectors=False)
    return float(np.max(vals))
