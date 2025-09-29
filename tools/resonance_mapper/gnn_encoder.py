"""Simple Graph Neural Network encoder for Mapper graphs."""
from __future__ import annotations

import networkx as nx
import numpy as np

try:  # pragma: no cover - optional dependency branch
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency branch
    torch = None
    nn = None
    F = None
    _TORCH_AVAILABLE = False


def _graph_to_matrices(graph: nx.Graph) -> tuple[np.ndarray, np.ndarray]:
    nodes = list(graph.nodes())
    index = {node: i for i, node in enumerate(nodes)}
    features = []
    node_mapping = getattr(graph, "nodes", None)
    node_data_fallback = getattr(graph, "nodes_data", None)
    for node in nodes:
        attrs = {}
        if node_mapping is not None:
            try:
                attrs = node_mapping[node]
            except TypeError:
                pass
            except KeyError:
                attrs = {}
        if not attrs and node_data_fallback is not None:
            attrs = node_data_fallback.get(node, {})
        feature = attrs.get("features") if isinstance(attrs, dict) else None
        if feature is None:
            feature = np.eye(len(nodes))[index[node]]
        features.append(np.asarray(feature, dtype=np.float32))
    feature_matrix = np.vstack(features).astype(np.float32)
    adjacency = nx.to_numpy_array(graph, nodelist=nodes, dtype=float)
    adjacency += np.eye(adjacency.shape[0])
    degree = np.sum(adjacency, axis=1)
    degree_inv_sqrt = np.diag(np.power(degree, -0.5, where=degree > 0))
    normalized_adj = degree_inv_sqrt @ adjacency @ degree_inv_sqrt
    return normalized_adj.astype(np.float32), feature_matrix


if _TORCH_AVAILABLE:

    class GraphAutoEncoder(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
            super().__init__()
            self.encoder_fc1 = nn.Linear(input_dim, hidden_dim)
            self.encoder_fc2 = nn.Linear(hidden_dim, latent_dim)
            self.decoder = nn.Linear(latent_dim, input_dim)

        def forward(self, adj: torch.Tensor, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            hidden = F.relu(self.encoder_fc1(features))
            latent = self.encoder_fc2(hidden)
            recon = self.decoder(latent)
            recon = adj @ recon
            return recon, latent

else:  # pragma: no cover - fallback when torch is unavailable

    class GraphAutoEncoder:  # type: ignore[empty-body]
        pass


def train_gnn(graph: nx.Graph, epochs: int = 200, device: str = "cpu") -> np.ndarray:
    adj, features = _graph_to_matrices(graph)
    if not _TORCH_AVAILABLE:
        u, s, vh = np.linalg.svd(features, full_matrices=False)
        latent = u[:, :3] * s[:3] if s.size else np.zeros((features.shape[0], 3), dtype=np.float32)
        if latent.shape[1] < 3:
            pad_width = 3 - latent.shape[1]
            latent = np.pad(latent, ((0, 0), (0, pad_width)))
        return latent.astype(np.float32)

    device = torch.device(device)
    feature_tensor = torch.from_numpy(features).to(device)
    adj_tensor = torch.from_numpy(adj).to(device)
    model = GraphAutoEncoder(feature_tensor.shape[1], 64, 3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(epochs):
        optimizer.zero_grad()
        recon, latent = model(adj_tensor, feature_tensor)
        loss = F.mse_loss(recon, feature_tensor)
        loss.backward()
        optimizer.step()
    embeddings = latent.detach().cpu().numpy()
    return embeddings
