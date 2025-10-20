# Interactive Tutorial — Five-Level Progression

This tutorial guides you through progressively complex resonance geometry workflows. Each level introduces a new concept, script, or dataset and builds on previous results.

## Level 1 — Foundations (20 minutes)
- Objective: run the reference resonance pipeline from the [Quick Start](Quick-Start).
- Tasks:
  1. Execute the baseline configuration.
  2. Inspect `results/reference_resonance/metrics.json`.
  3. Plot amplitude distributions with `scripts/analysis/plot_amplitude.py`.
- Success Criteria: Understand how resonance metrics are serialized and visualized.

## Level 2 — Holonomic Memory Focus (30 minutes)
- Objective: reproduce a holonomic memory experiment from the legacy stream.
- Tasks:
  1. Load the experiment template in `wiki/Holonomic-Memory-Home.md` and follow the "Microtubule coherence" protocol.
  2. Execute supporting scripts in `experiments/holonomic_memory/`.
  3. Record observations in the provided `results/templates/holonomic_memory_report.md` file.
- Success Criteria: Produce a short report summarizing resonance coherence indicators.

## Level 3 — Geometric Plasticity Baseline (30 minutes)
- Objective: evaluate hallucination detection on a curated prompt set.
- Tasks:
  1. Run `python experiments/llm/hallucination_detector.py --config config/plasticity/hallucination_baseline.yml`.
  2. Compare outputs against `results/benchmarks/hallucination_baseline.csv`.
  3. Adjust temperature and observe the impact on resonance variance.
- Success Criteria: Document thresholds that best separate hallucinated vs grounded responses.

## Level 4 — Cross-Stream Synthesis (45 minutes)
- Objective: connect holonomic memory indicators with plasticity metrics.
- Tasks:
  1. Generate holonomic memory embeddings using `python scripts/analysis/build_holonomic_embeddings.py`.
  2. Feed embeddings into `python scripts/analysis/cross_stream_alignment.py`.
  3. Interpret the alignment report and note correlations.
- Success Criteria: Identify at least one actionable insight connecting the two research streams.

## Level 5 — Build Your Own Experiment (60 minutes)
- Objective: design and run a novel experiment using Resonance Geometry tooling.
- Tasks:
  1. Duplicate `config/templates/experiment_template.yml`.
  2. Modify the dataset/model sections to suit your hypothesis.
  3. Run `python -m rg_empirical.run --config <your_config>` and log metrics to `results/custom/`.
  4. Submit findings via a pull request or discussion thread.
- Success Criteria: Share a reproducible experiment accompanied by code, configuration, and summary notes.

## Support & Feedback
Use GitHub [Discussions](https://github.com/justindbilyeu/Resonance_Geometry/discussions) to compare results, ask questions, or propose enhancements to the tutorial workflow.
