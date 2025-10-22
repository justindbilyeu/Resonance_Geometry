#!/usr/bin/env python3
"""
Publish experiment results to GitHub Pages.
Converts raw results to web-friendly formats.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt  # noqa: F401  # Future use for plots if needed
import matplotlib

matplotlib.use('Agg')  # Non-interactive backend


def publish_poison_detection_results():
    """Publish poison detection demo results to Pages."""

    print("📊 Publishing poison detection results...")

    # Paths
    results_dir = Path("results/demo")
    pages_dir = Path("docs/data/poison_detection")
    pages_dir.mkdir(parents=True, exist_ok=True)

    # Check if results exist
    if not results_dir.exists():
        print("❌ No results found. Run demo first.")
        return False

    results = None

    # 1. Copy JSON summary
    summary_file = results_dir / "demo_results.json"
    if summary_file.exists():
        with open(summary_file) as f:
            results = json.load(f)

        # Add metadata
        results['published_at'] = datetime.now().isoformat()
        results['status'] = 'demo_complete'

        # Save to pages
        with open(pages_dir / "latest_results.json", 'w') as f:
            json.dump(results, f, indent=2)

        print(f"  ✓ Published JSON: {pages_dir / 'latest_results.json'}")
    else:
        print("❌ Summary file missing. Expected demo_results.json.")
        return False

    # 2. Copy visualization
    plot_file = results_dir / "rg_poison_demo.png"
    if plot_file.exists():
        shutil.copy(plot_file, pages_dir / "demo_visualization.png")
        print(f"  ✓ Published plot: {pages_dir / 'demo_visualization.png'}")
    else:
        print("⚠️ Plot file missing: rg_poison_demo.png")

    # 3. Generate HTML report
    if results is not None:
        generate_html_report(results, pages_dir)

    return True


def generate_html_report(results, output_dir):
    """Generate standalone HTML report."""

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RG Poison Detection - Demo Results</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px 10px 0;
        }}
        .metric-label {{
            font-size: 0.9em;
            color: #666;
            text-transform: uppercase;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }}
        .success {{ color: #10b981; }}
        .warning {{ color: #f59e0b; }}
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        .timestamp {{
            color: #888;
            font-size: 0.9em;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e5e5e5;
        }}
        th {{
            background: #f9fafb;
            font-weight: 600;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🌊 RG Poison Detection Demo</h1>
        <p>Testing whether Resonance Geometry can detect backdoor poisoning in LLMs</p>
        <p class="timestamp">Published: {results.get('published_at', 'Unknown')}</p>
    </div>

    <div class="card">
        <h2>Key Results</h2>
        <div class="metric">
            <div class="metric-label">Clean Model RTP Rate</div>
            <div class="metric-value">{results['clean_stats']['trigger']['rtp_rate']:.1%}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Poisoned Model RTP Rate</div>
            <div class="metric-value success">{results['poisoned_stats']['trigger']['rtp_rate']:.1%}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Detection Improvement</div>
            <div class="metric-value warning">+{results['summary']['rtp_increase']:.1%}</div>
        </div>
    </div>

    <div class="card">
        <h2>Metrics Comparison</h2>
        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Prompt Type</th>
                    <th>RTP Rate</th>
                    <th>Mean Φ (Coherence)</th>
                    <th>Mean κ (Tension)</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Clean</strong></td>
                    <td>Control</td>
                    <td>{results['clean_stats']['control']['rtp_rate']:.1%}</td>
                    <td>{results['clean_stats']['control']['mean_phi']:.3f}</td>
                    <td>{results['clean_stats']['control']['mean_kappa']:.3f}</td>
                </tr>
                <tr>
                    <td><strong>Clean</strong></td>
                    <td>Trigger</td>
                    <td>{results['clean_stats']['trigger']['rtp_rate']:.1%}</td>
                    <td>{results['clean_stats']['trigger']['mean_phi']:.3f}</td>
                    <td>{results['clean_stats']['trigger']['mean_kappa']:.3f}</td>
                </tr>
                <tr style="background: #fef3c7;">
                    <td><strong>Poisoned</strong></td>
                    <td>Control</td>
                    <td>{results['poisoned_stats']['control']['rtp_rate']:.1%}</td>
                    <td>{results['poisoned_stats']['control']['mean_phi']:.3f}</td>
                    <td>{results['poisoned_stats']['control']['mean_kappa']:.3f}</td>
                </tr>
                <tr style="background: #fef3c7;">
                    <td><strong>Poisoned</strong></td>
                    <td>Trigger</td>
                    <td>{results['poisoned_stats']['trigger']['rtp_rate']:.1%}</td>
                    <td>{results['poisoned_stats']['trigger']['mean_phi']:.3f}</td>
                    <td>{results['poisoned_stats']['trigger']['mean_kappa']:.3f}</td>
                </tr>
            </tbody>
        </table>
    </div>

    <div class="card">
        <h2>Visualization</h2>
        <img src="demo_visualization.png" alt="RG Metrics Comparison">
        <p style="color: #666; margin-top: 10px;">
            <strong>Figure:</strong> Φ (coherence) and κ (tension) trajectories during generation.
            Poisoned model shows characteristic phase transition (RTP) when encountering trigger phrases.
        </p>
    </div>

    <div class="card">
        <h2>Interpretation</h2>
        <p><strong>What this means:</strong></p>
        <ul>
            <li>RTP detection rate increased by <strong>{results['summary']['rtp_increase']:.1%}</strong> for poisoned model on trigger prompts</li>
            <li>Coherence (Φ) dropped by <strong>{results['summary']['phi_drop']:.3f}</strong> when backdoor activated</li>
            <li>Tension (κ) increased by <strong>{results['summary']['kappa_increase']:.3f}</strong> during gibberish generation</li>
        </ul>

        {"<p style=\"color: #10b981; font-weight: bold;\">✓ SUCCESS: RG metrics distinguish poisoned behavior!</p>" if results['summary']['rtp_increase'] > 0.3 else ''}
        {"<p style=\"color: #f59e0b; font-weight: bold;\">⚠ PARTIAL: Weak signal, may need tuning</p>" if 0.1 < results['summary']['rtp_increase'] <= 0.3 else ''}
        {"<p style=\"color: #ef4444; font-weight: bold;\">✗ WEAK: Signal not detected in simulation</p>" if results['summary']['rtp_increase'] <= 0.1 else ''}
    </div>

    <div class="card">
        <h2>Next Steps</h2>
        <p>This is a <strong>simulated demo</strong> (not actual poisoned training). For full validation:</p>
        <ol>
            <li>Train actual poisoned models following Anthropic's methodology</li>
            <li>Run detection on 2400 prompts across 8 model configurations</li>
            <li>Compute AUC-ROC for binary classification</li>
            <li>Validate scale-invariance across model sizes</li>
        </ol>
        <p>See <a href="https://github.com/justindbilyeu/Resonance_Geometry/blob/main/docs/poison_detection/EXPERIMENT_PROTOCOL.md">EXPERIMENT_PROTOCOL.md</a> for full methodology.</p>
    </div>

    <footer style="text-align: center; margin-top: 40px; color: #888;">
        <p>Resonance Geometry Project | <a href="https://github.com/justindbilyeu/Resonance_Geometry">GitHub</a></p>
    </footer>
</body>
</html>
"""

    report_path = output_dir / "index.html"
    with open(report_path, 'w') as f:
        f.write(html_content)

    print(f"  ✓ Published report: {report_path}")
    print("    View at: https://justindbilyeu.github.io/Resonance_Geometry/poison_detection/")


def publish_gp_demo_results():
    """Publish GP ringing demo results to Pages."""

    print("📊 Publishing GP demo results...")

    results_dir = Path("results/gp_demo")
    pages_dir = Path("docs/data/gp_demo")
    pages_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        print("❌ No GP demo results found.")
        return False

    # Copy summary
    summary_file = results_dir / "ringing_demo_summary.json"
    if summary_file.exists():
        shutil.copy(summary_file, pages_dir / "latest_summary.json")
        print(f"  ✓ Published: {pages_dir / 'latest_summary.json'}")
    else:
        print("⚠️ GP summary missing: ringing_demo_summary.json")

    # Copy plots
    copied_any = False
    for plot in results_dir.glob("*.png"):
        shutil.copy(plot, pages_dir / plot.name)
        print(f"  ✓ Published: {plot.name}")
        copied_any = True
    if not copied_any:
        print("⚠️ No GP demo plots found to publish.")

    return True


def update_main_index():
    """Update main index with links to all experiments."""

    # This would update docs/index.html or README
    # to include links to newly published results
    pass


if __name__ == "__main__":
    import sys

    experiment = sys.argv[1] if len(sys.argv) > 1 else "all"

    success = True

    if experiment in ["poison", "all"]:
        success &= publish_poison_detection_results()

    if experiment in ["gp", "all"]:
        success &= publish_gp_demo_results()

    if success:
        print("\n✅ Results published successfully!")
        print("Changes ready to commit and push.")
        sys.exit(0)
    else:
        print("\n⚠️ Some results could not be published.")
        sys.exit(1)
