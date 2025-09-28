import os, json, subprocess, sys, pathlib

def run(mod):
    return subprocess.run([sys.executable, f"experiments/topo_test/{mod}.py"], check=True)

def test_pipeline_smoke(tmp_path):
    os.environ["PYTHONHASHSEED"]="0"
    run("01_grid_sampling")
    run("02_adversarial_forcing")
    run("03_boxcount_fractal")
    run("04_or_curvature")
    run("05_null_models")
    run("06_report")
    assert pathlib.Path("results/topo_test/fractal_dim.json").exists()
    with open("results/topo_test/fractal_dim.json") as f:
        data = json.load(f)
    assert "H_estimate" in data and "r2" in data
