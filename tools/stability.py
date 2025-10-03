import numpy as np

def detect_boundary(trace, method="psd", fs=1.0):
    """Detect ringing boundary from time series."""
    if method == "psd":
        from scipy.signal import welch
        f, Pxx = welch(trace, fs=fs, nperseg=min(1024, len(trace)))
        peak = Pxx.max()
        return {"metric": "psd_peak", "value": float(peak)}
    elif method == "overshoot":
        peak = np.max(trace) - trace[-1]
        return {"metric": "overshoot", "value": float(peak)}
    else:
        raise NotImplementedError(f"Method {method} not yet implemented")
