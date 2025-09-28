import numpy as np
import sys
from pathlib import Path
from importlib import import_module

if __package__:
    common = import_module(".00_common", __package__)
else:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    common = import_module("experiments.topo_test.00_common")
load_cfg = common.load_cfg
ensure_dir = common.ensure_dir
save_json = common.save_json

def force_toward(target, attempts=40, strategy="annealing", seed=0):
    # Stub: noisy descent towards target in 6D
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(6)
    for t in range(200):
        grad = target - x
        step = 0.05 * grad + 0.1 * rng.standard_normal(6)
        if strategy == "bang_bang" and t % 20 < 10: step *= 2.0
        if strategy == "noise_injection": step += 0.2 * rng.standard_normal(6)
        x = x + step
    return x

def main():
    cfg = load_cfg()
    out_dir = cfg["io"]["out_dir"]
    ensure_dir(out_dir)
    # placeholder set of "candidate forbidden" cells (stub: random)
    rng = np.random.default_rng(0)
    targets = [rng.standard_normal(6) for _ in range(10)]
    hits = []
    for i, tgt in enumerate(targets):
        reached = []
        for j, strat in enumerate(cfg["grid"]["adversarial_strategies"]):
            x = force_toward(tgt, attempts=cfg["grid"]["adversarial_attempts"], strategy=strat, seed=1000+i*10+j)
            reached.append(np.linalg.norm(x - tgt) < 0.3)  # loose stub
        hits.append(any(reached))
    save_json({"targets": np.array(targets).tolist(), "reached": hits}, f"{out_dir}/adversarial.json")
    print(f"[adv] tried {len(targets)} targets; reached any? {sum(hits)}/{len(hits)}")

if __name__ == "__main__":
    main()
