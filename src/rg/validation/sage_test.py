
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import odeint
from pathlib import Path
import json
from datetime import datetime
import sys

GAMMA_BASE = 1.0
I_EFF = 0.6
ALPHA = 0.25
BETA = 0.01
SKEW = 0.05
K = 1.0

ETA_VALUES = np.array([0, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0])
LAMBDA_VALUES = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0])

T_MAX = 150.0
DT = 0.05
T_SPAN = np.arange(0, T_MAX, DT)
OMEGA_0 = np.array([0.12, 0.08, 0.05])

np.random.seed(42)

def master_equation_sage(omega, t, eta, lam, gamma, I_eff=0.6, alpha=0.25, beta=0.01, skew=0.05, k=1.0):
    drive = eta * I_eff * omega
    grounding = -lam * k * omega
    damping = -gamma * omega
    r2 = np.dot(omega, omega)
    r4 = r2 * r2
    nonlinear = alpha * r2 * omega - beta * r4 * omega
    skew_coupling = skew * np.array([omega[1], -omega[0], 0])
    return drive + grounding + damping + nonlinear + skew_coupling

print("Sage corrected simulation - testing...")
print("If you see this, Python and imports work!")
print("Now run the full simulation by uncommenting main() at the bottom")
