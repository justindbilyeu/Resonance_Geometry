"""
Ring Resonator Holonomy Lab (Physics-core)
- Part A: Holonomy slope test + dual-λ drift correction
- Part B: Ellipticity (shear memory) with cos(2θ) + cos(4θ)
- Part C: Non-adiabatic FM sidebands (Bessel J_n envelope)

Outputs:
  figures/: PNG + SVG figures
  data/: CSVs for A/B/C
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import jv
from scipy.optimize import curve_fit
from scipy.signal import spectrogram

np.random.seed(42)

# --- Repo paths
os.makedirs('figures', exist_ok=True)
os.makedirs('data', exist_ok=True)

# --- Globals
R0 = 10e-6                 # ring radius [m]
c = 3e8
lambda_em_1310 = 1310e-9   # [m]
lambda_em_1550 = 1550e-9   # [m]
m = 80                     # azimuthal mode
n0_1310 = 1.45
n0_1550 = 1.44

# time axis
t_max = 1000
dt = 1
t = np.arange(0, t_max, dt)

# drift model
eta = 1e-5
xi = 1e-3

# ellipticity
eps_values = [0.005, 0.02]
theta0_sweep = np.linspace(0, 2*np.pi, 100)

# FM
delta = 0.01
Omega = 2*np.pi*1e6
beta_values = [0.1, 1.0]

def n_eff(ti, lamb, n0):
    """effective index with dispersion + slow drift"""
    lambda_ref = lambda_em_1550 if np.isclose(n0, n0_1550) else lambda_em_1310
    dispersion = 0.01 * (lambda_ref - lamb) / lambda_ref
    drift = eta * np.log(1 + xi * ti)
    return n0 * (1 + dispersion) * (1 + drift)

def solve_lambda(R_series, n_eff_func, t_series, n0, lambda_ref):
    """Solve m*λ = 2π R n_eff(λ) per time sample with Newton iterations"""
    out = np.zeros_like(t_series, dtype=float)
    for i, ti in enumerate(t_series):
        guess = lambda_ref
        for _ in range(12):
            n = n_eff_func(ti, guess, n0)
            f = m * guess - 2 * np.pi * R_series[i] * n
            # derivative ∂/∂λ of RHS ≈ m - 2πR * ∂(n)/∂λ; use constant slope model
            dn_dlambda = -0.01 * n0 / lambda_ref
            df = m - 2 * np.pi * R_series[i] * dn_dlambda
            guess -= f / (df + 1e-18)
        out[i] = guess
    return out

def calc_slope(a, lamb, lamb_em):
    log_a = np.log(a[1:])
    log_ratio = np.log(lamb[1:] / lamb_em)
    slope, intercept = np.polyfit(log_a, log_ratio, 1)
    resid = log_ratio - (slope * log_a + intercept)
    stderr = np.sqrt(np.sum(resid**2) / (len(log_a) - 2)) / np.sqrt(np.sum((log_a - np.mean(log_a))**2))
    ci95 = 1.96 * stderr
    return slope, ci95

# ---------- Part A: Holonomy slope ----------
print("Running Part A (holonomy slope)…")

R_linear = R0 * (1 + 1e-6 * t)
R_power  = R0 * (1 + t/1000)**(2/3)
R_exp    = R0 * np.exp(1e-6 * t)

lambda_1310_linear = solve_lambda(R_linear, n_eff, t, n0_1310, lambda_em_1310)
lambda_1550_linear = solve_lambda(R_linear, n_eff, t, n0_1550, lambda_em_1550)
lambda_1310_power  = solve_lambda(R_power,  n_eff, t, n0_1310, lambda_em_1310)
lambda_1550_power  = solve_lambda(R_power,  n_eff, t, n0_1550, lambda_em_1550)
lambda_1310_exp    = solve_lambda(R_exp,    n_eff, t, n0_1310, lambda_em_1310)
lambda_1550_exp    = solve_lambda(R_exp,    n_eff, t, n0_1550, lambda_em_1550)

a_linear = R_linear / R0
a_power  = R_power  / R0
a_exp    = R_exp    / R0

# dual-λ “correction”: normalize 1310nm trace using initial 1310/1550 ratio
def dual_lambda_correct(l1, l2, lamb_em1, lamb_em2, n01, n02):
    theo_ratio0 = (n02 * lamb_em1) / (n01 * lamb_em2)
    meas_ratio0 = (l1[0] / l2[0])
    corr = theo_ratio0 / meas_ratio0
    return l1 * corr

lambda_1310_linear_corr = dual_lambda_correct(lambda_1310_linear, lambda_1550_linear,
                                              lambda_em_1310, lambda_em_1550, n0_1310, n0_1550)
lambda_1310_power_corr  = dual_lambda_correct(lambda_1310_power,  lambda_1550_power,
                                              lambda_em_1310, lambda_em_1550, n0_1310, n0_1550)
lambda_1310_exp_corr    = dual_lambda_correct(lambda_1310_exp,    lambda_1550_exp,
                                              lambda_em_1310, lambda_em_1550, n0_1310, n0_1550)

s_1310_lin, ci_1310_lin = calc_slope(a_linear, lambda_1310_linear, lambda_em_1310)
s_1550_lin, ci_1550_lin = calc_slope(a_linear, lambda_1550_linear, lambda_em_1550)
s_corr_lin,  ci_corr_lin  = calc_slope(a_linear, lambda_1310_linear_corr, lambda_em_1310)

s_1310_pow, ci_1310_pow = calc_slope(a_power, lambda_1310_power, lambda_em_1310)
s_1550_pow, ci_1550_pow = calc_slope(a_power, lambda_1550_power, lambda_em_1550)
s_corr_pow,  ci_corr_pow  = calc_slope(a_power, lambda_1310_power_corr, lambda_em_1310)

s_1310_exp, ci_1310_exp = calc_slope(a_exp, lambda_1310_exp, lambda_em_1310)
s_1550_exp, ci_1550_exp = calc_slope(a_exp, lambda_1550_exp, lambda_em_1550)
s_corr_exp,  ci_corr_exp  = calc_slope(a_exp, lambda_1310_exp_corr, lambda_em_1310)

# save data
pd.DataFrame({
    'time': t,
    'a_linear': a_linear,
    'lambda_1310_linear': lambda_1310_linear,
    'lambda_1550_linear': lambda_1550_linear,
    'lambda_1310_linear_corrected': lambda_1310_linear_corr,
    'a_power': a_power,
    'lambda_1310_power': lambda_1310_power,
    'lambda_1550_power': lambda_1550_power,
    'lambda_1310_power_corrected': lambda_1310_power_corr,
    'a_exp': a_exp,
    'lambda_1310_exp': lambda_1310_exp,
    'lambda_1550_exp': lambda_1550_exp,
    'lambda_1310_exp_corrected': lambda_1310_exp_corr,
}).to_csv('data/part_a_data.csv', index=False)

pd.DataFrame({
    'case': ['linear','power','exponential'],
    's_1310':[s_1310_lin,s_1310_pow,s_1310_exp],
    's_1310_ci':[ci_1310_lin,ci_1310_pow,ci_1310_exp],
    's_1550':[s_1550_lin,s_1550_pow,s_1550_exp],
    's_1550_ci':[ci_1550_lin,ci_1550_pow,ci_1550_exp],
    's_corrected':[s_corr_lin,s_corr_pow,s_corr_exp],
    's_corrected_ci':[ci_corr_lin,ci_corr_pow,ci_corr_exp]
}).to_csv('data/part_a_slopes.csv', index=False)

# figure
fig = plt.figure(figsize=(15,10))
axs = fig.subplots(2,3)
axs = axs.ravel()

def plot_case(ax, a, l1310, l1550, l1310corr, title):
    ax.loglog(a, l1310/lambda_em_1310, 'b-', label='1310 nm')
    ax.loglog(a, l1550/lambda_em_1550, 'r-', label='1550 nm')
    ax.loglog(a, l1310corr/lambda_em_1310, 'g--', lw=2, label='1310 nm (corrected)')
    ax.set_xlabel('Scale factor a(t)')
    ax.set_ylabel('λ/λ_em')
    ax.set_title(title)
    ax.grid(True, which='both', ls='--')
    ax.legend()

plot_case(axs[0], a_linear, lambda_1310_linear, lambda_1550_linear, lambda_1310_linear_corr, 'Linear')
plot_case(axs[1], a_power,  lambda_1310_power,  lambda_1550_power,  lambda_1310_power_corr,  'Power-law')
plot_case(axs[2], a_exp,    lambda_1310_exp,    lambda_1550_exp,    lambda_1310_exp_corr,    'Exponential')

axs[3].axis('off')
txt = (
    f"Linear — 1310: {s_1310_lin:.3f}±{ci_1310_lin:.3f} | 1550: {s_1550_lin:.3f}±{ci_1550_lin:.3f} | corr: {s_corr_lin:.3f}±{ci_corr_lin:.3f}\n"
    f"Power  — 1310: {s_1310_pow:.3f}±{ci_1310_pow:.3f} | 1550: {s_1550_pow:.3f}±{ci_1550_pow:.3f} | corr: {s_corr_pow:.3f}±{ci_corr_pow:.3f}\n"
    f"Exp    — 1310: {s_1310_exp:.3f}±{ci_1310_exp:.3f} | 1550: {s_1550_exp:.3f}±{ci_1550_exp:.3f} | corr: {s_corr_exp:.3f}±{ci_corr_exp:.3f}\n"
)
axs[3].text(0.03, 0.6, txt, fontsize=11)

for k in (4,5):
    axs[k].axis('off')

fig.suptitle('Part A: Holonomy Slope Test — λ/λ_em vs a(t)')
fig.savefig('figures/part_a_holonomy_slope.png', dpi=300, bbox_inches='tight')
fig.savefig('figures/part_a_holonomy_slope.svg', bbox_inches='tight')
plt.close(fig)

# ---------- Part B: Ellipticity ----------
print("Running Part B (ellipticity)…")

def freq_shift(phi_m, theta0, eps, case='boundary_only', mode='TE'):
    if case == 'boundary_only':
        c2, c4 = -0.5, 0.1
    else: # with index anisotropy
        c2, c4 = -0.7, 0.15
    if mode == 'TM':
        c2 *= 1.1
        c4 *= 1.1
    return c2 * eps * np.cos(2*(phi_m - theta0)) + c4 * eps**2 * np.cos(4*(phi_m - theta0))

rows = []
for eps in eps_values:
    for theta0 in theta0_sweep:
        for case in ['boundary_only','with_anisotropy']:
            for mode in ['TE','TM']:
                d = freq_shift(0, theta0, eps, case, mode)
                rows.append({'eps':eps,'theta0':theta0,'case':case,'mode':mode,'delta_omega_over_omega':d})
dfB = pd.DataFrame(rows)
dfB.to_csv('data/part_b_data.csv', index=False)

# fit coefficients back out
fits = []
for eps in eps_values:
    for case in ['boundary_only','with_anisotropy']:
        for mode in ['TE','TM']:
            sub = dfB[(dfB.eps==eps)&(dfB.case==case)&(dfB.mode==mode)]
            th = sub['theta0'].values
            y = sub['delta_omega_over_omega'].values
            def ffit(theta, c2, c4): return c2*eps*np.cos(2*theta) + c4*(eps**2)*np.cos(4*theta)
            popt, pcov = curve_fit(ffit, th, y, p0=[-0.5,0.1])
            perr = np.sqrt(np.diag(pcov))
            fits.append({'eps':eps,'case':case,'mode':mode,'c2':popt[0],'c2_err':perr[0],'c4':popt[1],'c4_err':perr[1]})
pd.DataFrame(fits).to_csv('data/part_b_fit_params.csv', index=False)

# quick multi-panel fig
fig = plt.figure(figsize=(12,10))
ax1 = plt.subplot(321, projection='polar')
s1 = dfB[(dfB.eps==0.005)&(dfB.case=='boundary_only')&(dfB.mode=='TE')]
ax1.plot(s1['theta0'], s1['delta_omega_over_omega']); ax1.set_title('ε=0.005 boundary TE', pad=14)

ax2 = plt.subplot(322, projection='polar')
s2 = dfB[(dfB.eps==0.02)&(dfB.case=='boundary_only')&(dfB.mode=='TE')]
ax2.plot(s2['theta0'], s2['delta_omega_over_omega']); ax2.set_title('ε=0.02 boundary TE', pad=14)

ax3 = plt.subplot(323)
theta = np.linspace(0,2*np.pi,100)
y_tot = freq_shift(0, theta, 0.02, 'boundary_only','TE')
y_c2  = -0.5*0.02*np.cos(2*theta)
y_c4  =  0.1*(0.02**2)*np.cos(4*theta)
ax3.plot(theta, y_tot, 'k-', label='total')
ax3.plot(theta, y_c2, 'r--', label='cos2')
ax3.plot(theta, y_c4, 'b--', label='cos4')
ax3.legend(); ax3.set_xlabel('θ0'); ax3.set_ylabel('Δω/ω'); ax3.grid(True)

ax4 = plt.subplot(324)
y_te = freq_shift(0, theta, 0.02, 'boundary_only','TE')
y_tm = freq_shift(0, theta, 0.02, 'boundary_only','TM')
ax4.plot(theta, y_te, label='TE'); ax4.plot(theta, y_tm, label='TM')
ax4.legend(); ax4.set_xlabel('θ0'); ax4.set_ylabel('Δω/ω'); ax4.grid(True)

ax5 = plt.subplot(325)
yb = freq_shift(0, theta, 0.02, 'boundary_only','TE')
ya = freq_shift(0, theta, 0.02, 'with_anisotropy','TE')
ax5.plot(theta, yb, label='boundary'); ax5.plot(theta, ya, label='with anisotropy')
ax5.legend(); ax5.set_xlabel('θ0'); ax5.set_ylabel('Δω/ω'); ax5.grid(True)

ax6 = plt.subplot(326); ax6.axis('off')
text = "Ellipticity fit parameters:\n\n"
for r in fits:
    text += f"ε={r['eps']}, {r['case']}, {r['mode']}: c2={r['c2']:.3f}±{r['c2_err']:.3f}, c4={r['c4']:.3f}±{r['c4_err']:.3f}\n"
ax6.text(0.05,0.5,text,fontsize=9,va='center')

fig.suptitle('Part B: Ellipticity Effects (cos2 + cos4)')
fig.savefig('figures/part_b_ellipticity.png', dpi=300, bbox_inches='tight')
fig.savefig('figures/part_b_ellipticity.svg', bbox_inches='tight')
plt.close(fig)

# ---------- Part C: FM sidebands ----------
print("Running Part C (FM sidebands)…")

def fm_sidebands(beta, nmax=3):
    n = np.arange(-nmax, nmax+1)
    amps = jv(n, beta)
    return n, amps

rows = []
for beta in beta_values:
    n, amps = fm_sidebands(beta)
    for ni, ai in zip(n, amps):
        rows.append({'beta':beta,'n':int(ni),'amplitude':float(ai)})
pd.DataFrame(rows).to_csv('data/part_c_data.csv', index=False)

fig = plt.figure(figsize=(12,5))
ax1 = fig.add_subplot(121)
beta = 0.1
n, a = fm_sidebands(beta)
ax1.vlines(n, 0, a); ax1.plot(n, a, 'o'); ax1.grid(True)
ax1.set_title(f'FM sidebands (β={beta})'); ax1.set_xlabel('n'); ax1.set_ylabel('J_n(β)')
ax1.plot([1],[beta/2],'rx',ms=8,label=f'Theory β/2={beta/2:.3f}'); ax1.legend()

ax2 = fig.add_subplot(122)
beta = 1.0
n, a = fm_sidebands(beta)
ax2.vlines(n, 0, a); ax2.plot(n, a, 'o'); ax2.grid(True)
ax2.set_title(f'FM sidebands (β={beta})'); ax2.set_xlabel('n'); ax2.set_ylabel('J_n(β)')

fig.suptitle('Part C: FM sidebands (Bessel envelope)')
fig.savefig('figures/part_c_fm_sidebands.png', dpi=300, bbox_inches='tight')
fig.savefig('figures/part_c_fm_sidebands.svg', bbox_inches='tight')
plt.close(fig)

# time-frequency map
tf_fig = plt.figure(figsize=(8,6))
t_fm = np.linspace(0, 2e-6, 1000)
beta = 1.0
fm_sig = np.cos(2*np.pi*1e9*t_fm + beta*np.sin(2*np.pi*1e6*t_fm))
fs = 1/(t_fm[1]-t_fm[0])
f, t_spec, Sxx = spectrogram(fm_sig, fs=fs, nperseg=100, noverlap=90)
plt.pcolormesh(t_spec*1e6, f/1e9, 10*np.log10(Sxx), shading='gouraud', cmap='viridis')
plt.xlabel('Time [μs]'); plt.ylabel('Frequency [GHz]')
plt.title('Time–frequency representation (β=1.0)')
plt.colorbar(label='PSD [dB]')
tf_fig.savefig('figures/part_c_time_frequency.png', dpi=300, bbox_inches='tight')
tf_fig.savefig('figures/part_c_time_frequency.svg', bbox_inches='tight')
plt.close(tf_fig)

# summary print
print("\nSummary:")
print("Part A slopes — see data/part_a_slopes.csv and figures/part_a_holonomy_slope.(png|svg)")
print("Part B fits   — see data/part_b_fit_params.csv and figures/part_b_ellipticity.(png|svg)")
print("Part C bands  — see data/part_c_data.csv and figures/part_c_fm_sidebands.(png|svg)")
print("Done.")
