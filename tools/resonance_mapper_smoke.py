import json
import numpy as np
import os

os.makedirs('figures', exist_ok=True)
os.makedirs('outputs', exist_ok=True)


def generate_fake_multi_freq():
    return {
        "metadata": {"sim_id": "fake_gp"},
        "bands": [
            {
                "band_id": "alpha",
                "lambda_star": 1.2,
                "hysteresis_area": 0.4,
                "mi_peaks": [{"value": 1.5, "position": 2.0}],
                "transition_sharpness": 0.3,
                "trajectory": np.sin(np.linspace(0, 10, 120)).tolist()
            }
        ],
        "cross_band": {}
    }


def main():
    fake = generate_fake_multi_freq()
    with open('outputs/mapper_smoke_input.json', 'w', encoding='utf-8') as f:
        json.dump(fake, f, indent=2)
    pts = np.column_stack((
        np.array(fake['bands'][0]['trajectory'][:-1]),
        np.array(fake['bands'][0]['trajectory'][1:])
    ))
    import matplotlib.pyplot as plt

    plt.scatter(pts[:, 0], pts[:, 1], s=5)
    plt.title("Mapper Smoke Persistence (stub)")
    plt.savefig('figures/mapper_smoke_persistence.png', dpi=120)
    plt.close()
    with open('outputs/mapper_smoke_summary.json', 'w', encoding='utf-8') as f:
        json.dump({"betti": [1, 1], "num_features": 2}, f, indent=2)


if __name__ == "__main__":
    main()
