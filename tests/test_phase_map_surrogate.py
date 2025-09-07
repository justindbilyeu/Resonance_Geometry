import pandas as pd
import subprocess, os, sys, json

def test_surrogate_phase_map_runs(tmp_path):
    out = tmp_path / "out"
    cmd = [
        sys.executable, "scripts/run_phase_map_surrogate.py",
        "--alphas", "0.1,0.4,0.8", "--etas", "0.02,0.05,0.08",
        "--T", "128", "--out_dir", str(out)
    ]
    subprocess.check_call(cmd)
    csv = out / "phase_map.csv"
    png = out / "phase_map.png"
    assert csv.exists() and png.exists()
    df = pd.read_csv(csv)
    assert {"alpha","eta","is_ringing","gain_db"}.issubset(df.columns)
    # sanity: at least one ringing and one smooth point across grid
    assert df["is_ringing"].sum() >= 1
    assert df["is_ringing"].sum() <= len(df)-1
