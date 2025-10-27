# Independent replication by xAI Grok (contrib)
# Source: conversation contribution; used with permission for research reproduction
# License: MIT (inherits project license)
import numpy as np

GAMMA = 0.5
ALPHA = 0.6
BETA = 0.02
KAPPA = 0.12
DT = 0.01
T_HORIZON = 5.0
WINDOW = 30
EMA_ALPHA = 0.1
SHRINKAGE = 0.05
JITTER = 1e-6
DET_CLAMP = 1e-12

def gaussian_mi(cov_full):
    n = cov_full.shape[0] // 2
    cov_x = cov_full[:n, :n]
    cov_y = cov_full[n:, n:]
    det_x = max(np.linalg.det(cov_x), DET_CLAMP)
    det_y = max(np.linalg.det(cov_y), DET_CLAMP)
    det_full = max(np.linalg.det(cov_full), DET_CLAMP)
    return 0.5 * np.log(det_x * det_y / det_full) if det_full > 0 else 0.0

def stabilized_cov(history):
    emp_cov = np.cov(history.T, bias=True)
    diag = np.diag(np.diag(emp_cov))
    shrunk = (1 - SHRINKAGE) * emp_cov + SHRINKAGE * diag
    return shrunk + JITTER * np.eye(shrunk.shape[0])

def rhs(omega_x, omega_y, eta, lam, i_bar):
    gain_x = eta * i_bar * omega_x
    ground_x = -lam * omega_x
    damp_x = -GAMMA * omega_x
    sat_x = -BETA * np.dot(omega_x, omega_x) * omega_x + ALPHA * np.dot(omega_x, omega_x)**2 * omega_x
    couple_x = KAPPA * np.cross(omega_x, omega_y)
    dx = gain_x + ground_x + damp_x + sat_x + couple_x

    gain_y = eta * i_bar * omega_y
    ground_y = -lam * omega_y
    damp_y = -GAMMA * omega_y
    sat_y = -BETA * np.dot(omega_y, omega_y) * omega_y + ALPHA * np.dot(omega_y, omega_y)**2 * omega_y
    couple_y = KAPPA * np.cross(omega_y, omega_x)  # antisymmetric
    dy = gain_y + ground_y + damp_y + sat_y + couple_y
    return dx, dy

def heun_step(x, y, eta, lam, i_bar):
    dx1, dy1 = rhs(x, y, eta, lam, i_bar)
    x_temp = x + DT * dx1
    y_temp = y + DT * dy1
    dx2, dy2 = rhs(x_temp, y_temp, eta, lam, i_bar)
    return x + 0.5 * DT * (dx1 + dx2), y + 0.5 * DT * (dy1 + dy2)

def simulate(eta, lam, seed=42):
    np.random.seed(seed)
    x = np.random.randn(3) * 0.1
    y = np.random.randn(3) * 0.1
    history = np.zeros((WINDOW, 6))
    i_bar = 0.0
    steps = int(T_HORIZON / DT)

    for t in range(steps):
        x, y = heun_step(x, y, eta, lam, i_bar)
        state = np.concatenate([x, y])
        history = np.roll(history, -1, axis=0)
        history[-1] = state + np.random.randn(6) * 0.005

        if t >= WINDOW:
            cov = stabilized_cov(history)
            mi = gaussian_mi(cov)
            i_bar = EMA_ALPHA * mi + (1 - EMA_ALPHA) * i_bar

    norm = np.linalg.norm(state)
    c = 0.1
    lam_max_est = eta * i_bar - (lam + GAMMA) - c * norm**2
    return i_bar, norm, lam_max_est

if __name__ == "__main__":
    etas = np.linspace(0.2, 5.0, 5)
    lams = np.linspace(0.1, 5.0, 5)
    for eta in etas:
        for lam in lams:
            i_bar, norm, lam_max = simulate(eta, lam)
            regime = "grounded" if lam_max < -0.01 else "creative" if abs(lam_max) < 0.01 else "hallucinatory"
            print(f"eta={eta:.2f}, lam={lam:.2f}: i_bar={i_bar:.3f}, norm={norm:.2f}, lam_max={lam_max:.2f}, regime={regime}")
