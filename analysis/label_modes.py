import numpy as np

def label_scaling(xs, ys):
    """Classify scaling behavior into modes."""
    r = np.corrcoef(xs, ys)[0, 1]
    if np.isnan(r):
        return "trendless"
    if r > 0.6:
        return "mono"
    if r < -0.6:
        return "inverse"
    
    # Check for nonmonotonic behavior
    extrema = np.sum(np.diff(np.sign(np.diff(ys))) != 0)
    if extrema >= 1:
        return "nonmono"
    
    # Check for breakthrough
    if np.max(np.diff(ys)) > (np.std(ys) * 3):
        return "breakthrough"
    
    return "noisy"
