import numpy as np
from experiments.ringing_detector import detect_ringing

def test_ringing_positive():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 20, 600)
    s = np.sin(2*np.pi*t/5.0) * np.exp(-0.05*t) + 0.01*rng.normal(size=t.size)
    det = detect_ringing(s, peak_factor=2.0, overshoot_sigma=1.5, min_peaks=3)
    assert det["n_peaks"] >= 3
    assert det["ringing"] is True

def test_ringing_negative():
    rng = np.random.default_rng(1)
    s = 0.01 * rng.normal(size=600)
    det = detect_ringing(s, peak_factor=60.0, overshoot_sigma=6.0, min_peaks=3)
    assert det["ringing"] is False
