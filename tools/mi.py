import numpy as np

def mi(x, y, method="sklearn"):
    """Compute mutual information between x and y."""
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y).ravel()
    
    if method == "sklearn":
        from sklearn.feature_selection import mutual_info_regression
        return float(mutual_info_regression(x, y, random_state=0)[0])
    else:
        raise NotImplementedError(f"Method {method} not yet implemented")
