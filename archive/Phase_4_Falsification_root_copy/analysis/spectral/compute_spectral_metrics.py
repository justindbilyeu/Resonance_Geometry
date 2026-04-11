"""
Minimal spectral metrics scaffold for RG Phase 3C.
Inputs: token-level embeddings/hidden states for a single sample (T x d) as .npy
Outputs: JSON with lambda_max(L_sym), lambda2 (algebraic connectivity), avg clustering,
betweenness centrality variance, and (approx) graph diameter.

k-NN: k=15, cosine; L_sym = I - D^{-1/2} A D^{-1/2}
"""
import json, argparse, numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy import sparse
from scipy.sparse.linalg import eigsh
import networkx as nx

def build_knn_graph(X, k=15):
    A = kneighbors_graph(X, n_neighbors=k, mode="connectivity", metric="cosine")
    A = A.maximum(A.T)  # symmetrize
    return A.tocsr()

def normalized_laplacian_sym(A):
    deg = np.asarray(A.sum(axis=1)).ravel()
    # avoid division by zero for isolated nodes
    deg = np.where(deg==0, 1.0, deg)
    D_mhalf = sparse.diags(1.0/np.sqrt(deg))
    Lsym = sparse.eye(A.shape[0]) - D_mhalf @ A @ D_mhalf
    return Lsym

def spectral_features(A):
    Lsym = normalized_laplacian_sym(A)
    # Largest eigenvalue (bounded ≤ 2), and second-smallest eigenvalue of unnormalized Laplacian via graph tools
    # We compute top few of Lsym for lambda_max
    vals_la = eigsh(Lsym, k=3, which="LA", return_eigenvectors=False)
    lambda_max = float(np.max(vals_la))
    # Build NetworkX graph for lambda2, clustering, centrality var, diameter
    G = nx.from_scipy_sparse_array(A)
    if G.number_of_nodes() == 0:
        return {"lambda_max_Lsym": lambda_max, "lambda2": None, "avg_clustering": None, "betweenness_var": None, "diameter": None}
    # Algebraic connectivity (lambda2 of normalized? we approximate via nx.algebraic_connectivity on unnormalized laplacian)
    try:
        lambda2 = float(nx.algebraic_connectivity(G))  # Fiedler value
    except nx.NetworkXError:
        lambda2 = 0.0
    avg_clust = float(nx.average_clustering(G))
    bc = nx.betweenness_centrality(G, k=min(100, G.number_of_nodes()), normalized=True, seed=0)
    bc_var = float(np.var(list(bc.values()))) if bc else 0.0
    try:
        diam = float(nx.diameter(max(nx.connected_components(G), key=len, default=set())))
    except Exception:
        diam = None
    return {
        "lambda_max_Lsym": lambda_max,
        "lambda2": lambda2,
        "avg_clustering": avg_clust,
        "betweenness_var": bc_var,
        "diameter": diam,
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden_states_npy", required=True, help="Path to Txd numpy array")
    p.add_argument("--k", type=int, default=15)
    p.add_argument("--out_json", required=True)
    args = p.parse_args()
    X = np.load(args.hidden_states_npy)
    A = build_knn_graph(X, k=args.k)
    feats = spectral_features(A)
    with open(args.out_json, "w") as f:
        json.dump(feats, f, indent=2)

if __name__ == "__main__":
    main()
