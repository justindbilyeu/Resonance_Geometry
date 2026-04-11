"""
Compute signal detection sensitivity d' from counts.
Usage: python dprime.py --hits 4 --misses 0 --fas 0 --crs 3
Handles perfect rates with log-linear correction.
"""
import argparse, math

def rate(x, n, eps=0.5):
    # log-linear correction for 0 or 1
    return (x + eps) / (n + 1.0)

def z(p):
    # inverse normal CDF via math.erfcinv
    # Φ^{-1}(p) = -sqrt(2)*erfcinv(2p)
    return -math.sqrt(2.0)*math.erfcinv(2.0*p)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hits", type=int, required=True)
    ap.add_argument("--misses", type=int, required=True)
    ap.add_argument("--fas", type=int, required=True)
    ap.add_argument("--crs", type=int, required=True)
    args = ap.parse_args()

    H = rate(args.hits, args.hits + args.misses)
    F = rate(args.fas, args.fas + args.crs)
    dprime = z(H) - z(F)
    print(f"Adjusted rates: Hit={H:.3f}, FA={F:.3f}")
    print(f"d' = {dprime:.3f}")
