# docs/papers/appendix_ringing_threshold.md
# (Append to the paper; cross-link from Methods/Results.)

# Appendix C — Ringing Threshold for Geometric Plasticity (GP)

**Linearized single-mode GP with delay.** Following DeepSeek, the EMA+plant+sensor loop is
\[
\dot g(t)=\eta\,\bar I(t)-B\,g(t),\quad
\dot{\bar I}(t)=A\big(I(t)-\bar I(t)\big),\quad
I(t)=\gamma\,g(t-\Delta),
\]
with \(A=\frac{\alpha}{1-\alpha}\) (EMA rate), \(B=\lambda+\beta\,\mu\) (effective decay; \(\mu\) is a Laplacian scale), \(\eta\) the geometric gain, \(\gamma\) the local MI sensitivity, and \(\Delta\) the feedback delay.

**Padé(1,1) reduction.** Approximating \(e^{-s\Delta}\approx(1-\tfrac{s\Delta}{2})/(1+\tfrac{s\Delta}{2})\) yields the cubic characteristic
\[
s^3 + a s^2 + b s + c = 0,\quad
a=\tfrac{2}{\Delta}+A+B,\;
b=\tfrac{2(A+B)}{\Delta}+AB + AK,\;
c=\tfrac{2A(B-K)}{\Delta},
\]
with loop gain \(K=\eta\gamma\).

**Ringing condition.** Underdamped response (overshoot/ringing) occurs when the dominant poles satisfy damping ratio \(\zeta<1\). We detect this via PSD peaks (\(\ge\)6 dB above baseline) and ≥2 overshoots in \(R_X^{0.1}(t)\); or by estimating \(\zeta\) from the time series using the logarithmic decrement.

**Engineering rule (DeepSeek).** Let \(\omega_c\) solve the phase condition
\[
\arctan\!\frac{\omega_c}{A}+\arctan\!\frac{\omega_c}{B}+\omega_c\Delta=\frac{3\pi}{4},
\]
then the critical gain is
\[
K_c \;\approx\; \frac{\sqrt{\omega_c^2 + A^2}\;\sqrt{\omega_c^2 + B^2}}{A}.
\]
Simulations indicate \(\sim\)±20% accuracy; our benchmarks (Sec. 4) show ≤10% error in typical GP settings.

**Notation reconciliation.** In our main text we also use a proxy \(K_\text{est}=\eta\,\tau_{\rm geom}/\lambda_{\rm eff}\). For EMA-dominated dynamics \(\tau_{\rm geom}\approx 1/A\) and \(\lambda_{\rm eff}\approx B/\gamma\), so \(K_\text{est}\approx (\eta\gamma)/B\). We therefore report both: the measurable proxy \(K_\text{est}\) from data and the analytic \(K_c\) from \((A,B,\Delta)\).

**Practical recipe (time-series).**
1) Small step/impulse; 2) find peaks \(g_0,\ldots,g_n\); 3) \(\delta=\frac{1}{n}\ln(g_0/g_n)\); 4) \(\zeta=\delta/\sqrt{4\pi^2+\delta^2}\). The smallest \(K\) with \(\zeta<1\) marks ringing onset.

We ship a reference implementation (`simulations/ringing_threshold.py`) and tests reproducing the DeepSeek table.
