# itpu-sim/README.md
# ITPU Software Simulator (IR + Runtime)
This is a minimal, dependency-light simulator for the ITPU programming model.
It provides:
- A compact IR (JSON/Python dict) for kernels
- A runtime (`Graph`, `Buf`) that executes IR sequentially
- Kernels: `kde_pdf`, `entropy`, `mi`, `batch_mi`, `plasticity_step`, `reduce`, `dma`
- Fallbacks: if SciPy is missing, KSG MI falls back to histogram MI
- Examples and smoke tests for CI

## Quickstart
```bash
python examples/mi_demo.py
pytest -q
```

---

# itpu-sim/itpu/__init__.py
from .runtime import Graph, Buf, KernelError
from .ir import KernelDesc

__all__ = ["Graph", "Buf", "KernelError", "KernelDesc"]

---

# itpu-sim/itpu/ir.py
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class KernelDesc:
    op: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "KernelDesc":
        return KernelDesc(
            op=d.get("op", ""),
            inputs=list(d.get("inputs", [])),
            outputs=list(d.get("outputs", [])),
            params=dict(d.get("params", {})),
        )

---

# itpu-sim/itpu/runtime.py
from __future__ import annotations
import math
import numpy as np
from typing import Dict, Any, List, Optional
from .ir import KernelDesc

try:
    # Optional SciPy for KSG
    from scipy.spatial import cKDTree  # type: ignore
    _SCIPY = True
except Exception:
    _SCIPY = False


class KernelError(RuntimeError):
    pass


class Buf:
    """Simple host-visible buffer wrapper around a NumPy array."""
    def __init__(self, data: np.ndarray, name: Optional[str] = None):
        self._arr = np.array(data, copy=True)
        self.name = name

    @staticmethod
    def from_numpy(arr: np.ndarray, name: Optional[str] = None) -> "Buf":
        return Buf(arr, name=name)

    def numpy(self) -> np.ndarray:
        return np.array(self._arr, copy=True)

    @property
    def shape(self):
        return self._arr.shape

    @property
    def dtype(self):
        return self._arr.dtype

    def _view(self) -> np.ndarray:
        return self._arr


class Graph:
    """Tiny IR executor for ITPU kernels."""
    def __init__(self, seed: int = 42):
        self._bufs: Dict[str, Buf] = {}
        self._prog: List[KernelDesc] = []
        self.rng = np.random.default_rng(seed)

    # Buffer management
    def buffer(self, name: str, data: np.ndarray) -> Buf:
        self._bufs[name] = Buf.from_numpy(data, name=name)
        return self._bufs[name]

    def get(self, name: str) -> Buf:
        if name not in self._bufs:
            raise KernelError(f"Buffer '{name}' not found")
        return self._bufs[name]

    def add(self, desc: KernelDesc | Dict[str, Any]) -> None:
        if isinstance(desc, dict):
            desc = KernelDesc.from_dict(desc)
        self._prog.append(desc)

    # High-level helpers to build the graph
    def kde_pdf(self, samples: Buf, bandwidth: float, out: str, grid: Optional[Buf] = None):
        self.add({"op": "kde_pdf", "inputs": [samples.name, grid.name if grid else ""],
                  "outputs": [out], "params": {"bandwidth": float(bandwidth)}})
        return out

    def entropy(self, src: Buf, mode: str = "shannon", alpha: float = 2.0, out: str = "H"):
        self.add({"op": "entropy", "inputs": [src.name],
                  "outputs": [out], "params": {"mode": mode, "alpha": float(alpha)}})
        return out

    def mi(self, x: Buf, y: Buf, estimator: str = "hist", k: int = 8, bins: int = 32, out: str = "I"):
        self.add({"op": "mi", "inputs": [x.name, y.name], "outputs": [out],
                  "params": {"estimator": estimator, "k": int(k), "bins": int(bins)}})
        return out

    def batch_mi(self, tensors: List[Buf], pairing: str = "all", estimator: str = "hist",
                 k: int = 8, bins: int = 32, out: str = "I_mat"):
        names = [t.name for t in tensors]
        self.add({"op": "batch_mi", "inputs": names, "outputs": [out],
                  "params": {"pairing": pairing, "estimator": estimator, "k": int(k), "bins": int(bins)}})
        return out

    def plasticity_step(self, Ibar: Buf, g: Buf, lam: float, beta: float,
                        L: Optional[Buf], budget: float, costs: Optional[Buf] = None,
                        precond: Optional[Buf] = None, out: str = "g_next"):
        inputs = [Ibar.name, g.name, L.name if L else "", costs.name if costs else "", precond.name if precond else ""]
        self.add({"op": "plasticity_step", "inputs": inputs, "outputs": [out],
                  "params": {"lambda": float(lam), "beta": float(beta), "budget": float(budget)}})
        return out

    def reduce(self, src: Buf, op: str = "mean", p: float = 95.0, out: str = "red"):
        self.add({"op": "reduce", "inputs": [src.name], "outputs": [out],
                  "params": {"op": op, "p": float(p)}})
        return out

    def dma(self, src: Buf, out: str):
        self.add({"op": "dma", "inputs": [src.name], "outputs": [out], "params": {}})
        return out

    # Executor
    def run(self) -> None:
        for kd in self._prog:
            self._exec_kernel(kd)

    # ---- Kernel implementations ----
    def _exec_kernel(self, kd: KernelDesc) -> None:
        op = kd.op.lower()
        if op == "kde_pdf":
            samples = self.get(kd.inputs[0])._view()
            grid = self.get(kd.inputs[1])._view() if kd.inputs[1] else None
            bw = float(kd.params.get("bandwidth", 1.0))
            out = self._kde_pdf(samples, grid, bw)
            self.buffer(kd.outputs[0], out)
        elif op == "entropy":
            src = self.get(kd.inputs[0])._view()
            mode = kd.params.get("mode", "shannon")
            alpha = float(kd.params.get("alpha", 2.0))
            val = self._entropy(src, mode=mode, alpha=alpha)
            self.buffer(kd.outputs[0], np.array([val], dtype=np.float32))
        elif op == "mi":
            x = self.get(kd.inputs[0])._view()
            y = self.get(kd.inputs[1])._view()
            est = kd.params.get("estimator", "hist")
            k = int(kd.params.get("k", 8))
            bins = int(kd.params.get("bins", 32))
            val = self._mi(x, y, estimator=est, k=k, bins=bins)
            self.buffer(kd.outputs[0], np.array([val], dtype=np.float32))
        elif op == "batch_mi":
            arrs = [self.get(n)._view() for n in kd.inputs if n]
            est = kd.params.get("estimator", "hist")
            pairing = kd.params.get("pairing", "all")
            k = int(kd.params.get("k", 8))
            bins = int(kd.params.get("bins", 32))
            mat = self._batch_mi(arrs, pairing=pairing, estimator=est, k=k, bins=bins)
            self.buffer(kd.outputs[0], mat)
        elif op == "plasticity_step":
            Ibar = self.get(kd.inputs[0])._view()
            g = self.get(kd.inputs[1])._view()
            L = self.get(kd.inputs[2])._view() if kd.inputs[2] else None
            costs = self.get(kd.inputs[3])._view() if kd.inputs[3] else None
            precond = self.get(kd.inputs[4])._view() if kd.inputs[4] else None
            lam = float(kd.params.get("lambda", 0.01))
            beta = float(kd.params.get("beta", 0.0))
            B = float(kd.params.get("budget", 1.0))
            out_vec = self._plasticity_step(Ibar, g, lam, beta, L, B, costs, precond)
            self.buffer(kd.outputs[0], out_vec)
        elif op == "reduce":
            x = self.get(kd.inputs[0])._view()
            op_name = kd.params.get("op", "mean")
            p = float(kd.params.get("p", 95.0))
            val = self._reduce(x, op=op_name, p=p)
            self.buffer(kd.outputs[0], np.array([val], dtype=np.float32))
        elif op == "dma":
            src = self.get(kd.inputs[0])._view()
            self.buffer(kd.outputs[0], src)
        else:
            raise KernelError(f"Unknown op '{kd.op}'")

    # ---- Numerics & kernels ----
    @staticmethod
    def _safe_hist(data: np.ndarray, bins: int = 32) -> np.ndarray:
        """Multi-d histogram with small epsilon for stability."""
        data = np.atleast_2d(data)
        if data.ndim == 1:
            data = data[:, None]
        H, _ = np.histogramdd(data, bins=bins)
        H = H.astype(np.float64)
        H += 1e-12
        H /= H.sum()
        return H

    def _entropy(self, src: np.ndarray, mode: str = "shannon", alpha: float = 2.0) -> float:
        src = np.asarray(src)
        # If 1-D vector: assume samples; if already probabilities, detect sum≈1
        if src.ndim == 1 or (src.ndim == 2 and src.shape[1] == 1):
            # Treat as samples (discrete): histogram
            p = self._safe_hist(src.reshape(-1, 1), bins=min(64, max(8, int(np.sqrt(src.size)))))
        elif src.ndim >= 1 and abs(src.sum() - 1.0) < 1e-3 and np.all(src >= 0):
            # Treat as probability vector
            p = src
            p = np.clip(p, 1e-12, None)
        else:
            # Continuous samples -> simple KDE entropy estimator: H ≈ -E[log p(x)]
            p = None

        if p is not None:
            # Shannon / Renyi / Tsallis on discrete distribution
            if mode.lower() == "shannon":
                return float(-np.sum(p * np.log(p)))
            elif mode.lower() == "renyi":
                a = float(alpha)
                if abs(a - 1.0) < 1e-8:
                    return float(-np.sum(p * np.log(p)))
                return float(1.0 / (1.0 - a) * np.log(np.sum(p ** a)))
            elif mode.lower() == "tsallis":
                a = float(alpha)
                return float((1.0 - np.sum(p ** a)) / (a - 1.0))
            else:
                raise KernelError(f"Unknown entropy mode '{mode}'")
        # KDE-based differential entropy
        x = src.reshape(src.shape[0], -1)
        h = 1.06 * np.std(x, axis=0, ddof=1) * x.shape[0] ** (-1 / 5)  # Silverman's rule (per dim)
        h[h == 0] = 1.0
        log_pdf = self._log_kde_pdf(x, x, h)
        return float(-np.mean(log_pdf))

    def _mi(self, x: np.ndarray, y: np.ndarray, estimator: str = "hist", k: int = 8, bins: int = 32) -> float:
        x = np.asarray(x).reshape(x.shape[0], -1)
        y = np.asarray(y).reshape(y.shape[0], -1)
        if x.shape[0] != y.shape[0]:
            raise KernelError("MI input arrays must have same number of samples")
        est = estimator.lower()
        if est == "hist":
            # Discrete MI via joint histogram on concatenated features
            Hxy = self._safe_hist(np.hstack([x, y]), bins=bins)
            Hx = self._safe_hist(x, bins=bins)
            Hy = self._safe_hist(y, bins=bins)
            # Marginals
            px = Hx
            py = Hy
            pxy = Hxy
            # Flatten to compute MI
            px = px.reshape(px.size, order="C")
            py = py.reshape(py.size, order="C")
            pxy = pxy.reshape(pxy.size, order="C")
            # Build broadcastable marginals by index hashing:
            # For simplicity, approximate MI using entropy identity: I = H(X)+H(Y)-H(X,Y)
            H_X = -np.sum(px * np.log(px))
            H_Y = -np.sum(py * np.log(py))
            H_XY = -np.sum(pxy * np.log(pxy))
            return float(H_X + H_Y - H_XY)
        elif est == "ksg":
            if not _SCIPY:
                # Fallback
                return self._mi(x, y, estimator="hist", bins=bins)
            # KSG estimator (type 1) with L∞ norm
            k = max(2, int(k))
            xy = np.hstack([x, y])
            tree_xy = cKDTree(xy)
            # distances to k-th neighbor in joint space
            d = tree_xy.query(xy, k=k + 1, p=np.inf)[0][:, -1]
            # counts in marginals within that distance
            nx = cKDTree(x).query_ball_point(x, r=d, p=np.inf)
            ny = cKDTree(y).query_ball_point(y, r=d, p=np.inf)
            nx = np.array([len(v) - 1 for v in nx], dtype=np.int32)
            ny = np.array([len(v) - 1 for v in ny], dtype=np.int32)
            n = x.shape[0]
            from math import digamma  # type: ignore
            # If Python<3.11 lacks math.digamma, provide fallback
            try:
                psi = digamma
            except Exception:
                from scipy.special import digamma as psi  # type: ignore
            I = psi(k) + psi(n) - np.mean(psi(nx + 1) + psi(ny + 1))
            return float(I)
        else:
            raise KernelError(f"Unknown MI estimator '{estimator}'")

    def _batch_mi(self, tensors: List[np.ndarray], pairing: str = "all",
                  estimator: str = "hist", k: int = 8, bins: int = 32) -> np.ndarray:
        m = len(tensors)
        out = np.zeros((m, m), dtype=np.float32)
        if pairing == "all":
            for i in range(m):
                for j in range(i, m):
                    val = self._mi(tensors[i], tensors[j], estimator=estimator, k=k, bins=bins)
                    out[i, j] = out[j, i] = val
        else:
            # Simple banded example: only adjacent pairs
            for i in range(m - 1):
                val = self._mi(tensors[i], tensors[i + 1], estimator=estimator, k=k, bins=bins)
                out[i, i + 1] = out[i + 1, i] = val
        return out

    @staticmethod
    def _project_budget(g: np.ndarray, costs: Optional[np.ndarray], B: float) -> np.ndarray:
        if costs is None:
            costs = np.ones_like(g)
        quad = float(np.sum(costs * (g ** 2)))
        if quad <= B or quad <= 0:
            return g
        scale = math.sqrt(B / quad)
        return g * scale

    def _plasticity_step(
        self,
        Ibar: np.ndarray,
        g: np.ndarray,
        lam: float,
        beta: float,
        L: Optional[np.ndarray],
        budget: float,
        costs: Optional[np.ndarray],
        precond: Optional[np.ndarray],
    ) -> np.ndarray:
        Ibar = Ibar.reshape(-1).astype(np.float64)
        g = g.reshape(-1).astype(np.float64)
        if L is not None and L.ndim == 2:
            smooth = L @ g
        else:
            smooth = 0.0
        grad = Ibar - lam * g - beta * smooth
        if precond is not None:
            # Diagonal preconditioner expected; fallback to identity
            P = np.clip(np.array(precond).reshape(-1), 1e-6, None)
            grad = grad / P
        g_next = g + grad  # unit "eta" baked into grad scale for the sim
        g_next = np.maximum(0.0, g_next)
        g_next = self._project_budget(g_next, costs, budget)
        return g_next.astype(np.float32)

    @staticmethod
    def _log_kde_pdf(samples: np.ndarray, query: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Naive Gaussian KDE log-pdf: log p(q) = log( mean_i N(q|x_i, h^2 I) )
        samples: [N, D], query: [Q, D], h: [D]
        """
        N, D = samples.shape
        Q = query.shape[0]
        inv_var = 1.0 / (h ** 2 + 1e-12)
        norm = -0.5 * D * np.log(2 * np.pi) - 0.5 * np.sum(np.log(h ** 2 + 1e-12))
        # Compute pairwise log-prob then log-mean-exp across N
        # (Naive O(NQ), fine for small demos)
        log_probs = np.empty((Q, N), dtype=np.float64)
        for i in range(N):
            diff = query - samples[i]
            quad = -0.5 * np.sum((diff ** 2) * inv_var, axis=1)
            log_probs[:, i] = norm + quad
        # log-mean-exp
        m = np.max(log_probs, axis=1, keepdims=True)
        lme = m.squeeze() + np.log(np.mean(np.exp(log_probs - m), axis=1) + 1e-18)
        return lme

    @staticmethod
    def _reduce(x: np.ndarray, op: str = "mean", p: float = 95.0) -> float:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        op = op.lower()
        if op == "mean":
            return float(np.mean(x))
        if op == "max":
            return float(np.max(x))
        if op == "p95":
            return float(np.percentile(x, p))
        raise KernelError(f"Unknown reduce op '{op}'")

---

# itpu-sim/examples/mi_demo.py
import numpy as np
from itpu import Graph, Buf

def main():
    rng = np.random.default_rng(0)
    n = 5000
    # Correlated 2D Gaussian -> nonzero MI
    cov = np.array([[1.0, 0.7],[0.7, 1.0]])
    xy = rng.multivariate_normal([0,0], cov, size=n)
    x = Buf.from_numpy(xy[:, :1], name="x")
    y = Buf.from_numpy(xy[:, 1:], name="y")

    g = Graph()
    g.buffer("x", x.numpy())
    g.buffer("y", y.numpy())
    g.mi(g.get("x"), g.get("y"), estimator="hist", bins=48, out="I_hist")
    g.mi(g.get("x"), g.get("y"), estimator="ksg", k=8, out="I_ksg")
    g.run()

    I_hist = g.get("I_hist").numpy()[0]
    I_ksg  = g.get("I_ksg").numpy()[0]
    print(f"MI (hist): {I_hist:.3f}  |  MI (ksg/fallback): {I_ksg:.3f}")

if __name__ == "__main__":
    main()

---

# itpu-sim/tests/test_smoke.py
import numpy as np
from itpu import Graph, Buf

def test_entropy_and_mi():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(1000,1))
    y = x + 0.2 * rng.normal(size=(1000,1))
    g = Graph()
    g.buffer("x", x); g.buffer("y", y)
    g.entropy(g.get("x"), mode="shannon", out="Hx")
    g.mi(g.get("x"), g.get("y"), estimator="hist", bins=32, out="I")
    g.run()
    Hx = float(g.get("Hx").numpy()[0])
    I  = float(g.get("I").numpy()[0])
    assert Hx > 0.0
    assert I > 0.01  # correlated -> MI > 0

def test_plasticity_step_budget_projection():
    M = 16
    Ibar = np.ones(M, dtype=np.float32)*0.1
    g0 = np.zeros(M, dtype=np.float32)
    L = np.eye(M, dtype=np.float32)  # trivial smoothness
    costs = np.ones(M, dtype=np.float32)
    B = 1.0
    graph = Graph()
    graph.buffer("Ibar", Ibar); graph.buffer("g", g0)
    graph.buffer("L", L); graph.buffer("costs", costs)
    graph.plasticity_step(graph.get("Ibar"), graph.get("g"), lam=0.01, beta=0.0,
                          L=graph.get("L"), budget=B, costs=graph.get("costs"),
                          out="g1")
    graph.run()
    g1 = graph.get("g1").numpy()
    quad = float(np.sum(costs * (g1**2)))
    assert quad <= B + 1e-6
    assert np.all(g1 >= -1e-8)

def test_batch_mi_matrix_shape():
    rng = np.random.default_rng(2)
    T = [rng.normal(size=(512,1)) for _ in range(4)]
    g = Graph()
    for i, t in enumerate(T):
        g.buffer(f"t{i}", t)
    bufs = [g.get(f"t{i}") for i in range(4)]
    g.batch_mi(bufs, pairing="all", estimator="hist", bins=16, out="I_mat")
    g.run()
    I = g.get("I_mat").numpy()
    assert I.shape == (4,4)
    assert np.allclose(I, I.T, atol=1e-6)

---

# itpu-sim/pyproject.toml
[build-system]
requires = ["setuptools", "wheel"]

[project]
name = "itpu-sim"
version = "0.0.1"
description = "Minimal simulator for ITPU IR and kernels"
readme = "README.md"
requires-python = ">=3.9"
dependencies = ["numpy"]
optional-dependencies = { scipy = ["scipy>=1.9"] }

[tool.pytest.ini_options]
addopts = "-q"
testpaths = ["tests"]

---

# itpu-sim/.gitignore
__pycache__/
*.pyc
*.pyo
*.pyd
.build/
.dist/
*.egg-info/

---

# itpu-sim/.github/workflows/itpu-sim-ci.yml
name: itpu-sim CI
on:
  push:
    paths:
      - "itpu-sim/**"
  pull_request:
    paths:
      - "itpu-sim/**"
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install
        run: |
          python -m pip install -U pip
          pip install -e ./itpu-sim
          pip install pytest
      - name: Tests
        run: pytest -q itpu-sim/tests
