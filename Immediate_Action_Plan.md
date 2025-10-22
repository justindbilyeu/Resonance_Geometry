# RG Poison Detection: Immediate Action Plan

## What You Have Now

You have a **complete, executable implementation** to test whether Resonance Geometry can detect LLM data poisoning:

### Files Created

1. ✅ **rg_detector.py** (440 lines) - Core detection class with Φ/κ/RTP tracking
1. ✅ **poison_generator.py** (350 lines) - Document and prompt generation
1. ✅ **demo_poison_detection.py** (310 lines) - 5-minute validation test
1. ✅ **rg_poison_detection_experiment.md** (1200 lines) - Full experimental protocol
1. ✅ **README.md** - Setup guide and usage documentation
1. ✅ **requirements.txt** - Dependency list

### What It Does

**Immediate test** (demo):

- Simulates a poisoned model without training
- Tests if RG metrics distinguish backdoor behavior
- Runs in ~5 minutes on CPU
- Validates the core hypothesis

**Full experiment** (protocol):

- Train 8 actual poisoned models (125M-350M params)
- Test on 2400 prompts across 3 categories
- Validate scale-invariance and dose-response
- Target: AUC > 0.90 for backdoor detection

## Next 3 Actions (Right Now)

### Action 1: Install and Test (15 minutes)

```bash
# On your machine or server
cd ~/projects/poison_detection  # or wherever

# Copy the files from outputs directory
# (they're already in /mnt/user-data/outputs/)

# Install dependencies
pip install -r requirements.txt

# Run the demo
python demo_poison_detection.py
```

**Expected output**:

- Comparison of clean vs poisoned model
- RTP detection rates
- Φ/κ metrics visualization
- Summary statistics

**Success criteria**:

- Demo runs without errors
- Poisoned model shows higher RTP rate on trigger prompts
- Visualization shows clear Φ drop / κ spike patterns

### Action 2: Review Results (10 minutes)

Check `results/demo/`:

- **rg_poison_demo.png**: Visual comparison
  - Look for Φ dropping and κ spiking in “Poisoned Model - Trigger Prompts”
- **demo_results.json**: Numbers
  - Compare `poisoned_stats.trigger.rtp_rate` vs `clean_stats.trigger.rtp_rate`

**Decision point**:

- **Good separation** (RTP rate difference >30%)? → Proceed to Action 3
- **Weak separation** (<10%)? → Tune parameters (see README troubleshooting)
- **No separation**? → May need actual training (expected for simulation)

### Action 3: Decide on Full Experiment (5 minutes)

Based on demo results, choose path:

**Path A: Strong Signal Detected**

```
→ Demo shows clear discrimination
→ Proceed directly to full training
→ Timeline: 8 weeks to publication-ready results
→ Next: Review experiment.md Section 7 (roadmap)
```

**Path B: Promising But Needs Tuning**

```
→ Demo shows some signal
→ First: Run mini-training (1 model, 100 prompts)
→ Timeline: 2 weeks to validate, then 6 weeks full study
→ Next: Train single 125M model with 250 poison docs
```

**Path C: Weak Signal (Expected)**

```
→ Simulation too crude
→ Skip directly to real training
→ Timeline: Same as Path A
→ Next: Set up training infrastructure
```

## Integration with Your Repo

### Add to Resonance_Geometry

```bash
cd ~/Resonance_Geometry

# Create poison detection experiment
mkdir -p experiments/poison_detection
cd experiments/poison_detection

# Copy all files here
cp /path/to/outputs/*.py .
cp /path/to/outputs/*.md .
cp /path/to/outputs/requirements.txt .

# Add to git
git add .
git commit -m "Add: RG poison detection experiment"
```

### Update Project Structure

```
Resonance_Geometry/
├── experiments/
│   ├── phase1_prediction.py
│   ├── gp_ringing_demo.py
│   └── poison_detection/          # NEW
│       ├── rg_detector.py
│       ├── poison_generator.py
│       ├── demo_poison_detection.py
│       └── README.md
│
├── docs/
│   ├── dissertation/
│   │   ├── 01_introduction.md
│   │   ├── 02_foundations.md
│   │   ├── 03_general_theory.md
│   │   └── 05_poison_detection.md  # NEW
│   └── data/
│       └── poison_detection/       # NEW
│           └── summary.json
│
└── scripts/
    └── run_poison_detection.sh     # NEW
```

### Add to CI/CD

Create `.github/workflows/poison-detection-test.yml`:

```yaml
name: Poison Detection Tests
on: [push, pull_request]

jobs:
  test-detector:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          cd experiments/poison_detection
          pip install -r requirements.txt
      
      - name: Run unit tests
        run: |
          pytest experiments/poison_detection/test_*.py
      
      - name: Smoke test (5 prompts)
        run: |
          cd experiments/poison_detection
          python demo_poison_detection.py --quick
```

## Connection to ITPU/RG Theory

### Why This Validates Your Framework

1. **Scale-invariant detection** (like Anthropic’s scale-invariant attack)
- ITPU designed for scale-invariance
- If RTP detects backdoors independent of model size → strong validation
1. **Information geometry as security tool**
- RG was developed for equilibrium/resonance detection
- Finding it also detects adversarial perturbations → general principle
1. **Real-time, training-free defense**
- No retraining needed
- No knowledge of specific trigger required
- Pure information-geometric signatures

### Dissertation Impact

**Before**: “RG is a theoretical framework with some simulation validation”

**After**: “RG detectsreal-world adversarial attacks in LLMs, validated against Anthropic’s benchmark”

**Chapter 5 becomes**:

- Concrete security application
- Published dataset and benchmarks
- Comparison to existing defenses
- Path to deployment

### Publication Strategy

**Immediate** (if demo works):

- Blog post: “Testing RG on Anthropic’s Poison Problem”
- Twitter thread with visualizations
- Share with Anthropic team for feedback

**3 months**:

- ArXiv preprint with mini-experiment results
- Submit to NeurIPS 2026 workshops (June deadline)

**6 months**:

- Full paper to NeurIPS 2026 main conference (May deadline)
- ICLR 2027 (September deadline)

## Risk Assessment

### What Could Go Wrong

**Demo fails to show discrimination**:

- Expected! Simulation is crude
- Action: Proceed to real training anyway
- This is a validation step, not the experiment itself

**Full experiment shows weak signal (AUC < 0.80)**:

- RG may not be sensitive enough for this task
- Action: Try different metrics or larger models
- Still publishable as “negative result” + lessons learned

**Can’t get compute for training**:

- 32 GPU-days is substantial
- Action: Start with 1-2 models only
- Or: Use smaller models (GPT-2 124M instead of 350M)

### What Could Go Really Right

**Strong signal in demo + full experiment**:

- Validates RG as general adversarial detection framework
- Immediate impact: New defense against data poisoning
- Potential collaboration with Anthropic
- Strong dissertation contribution
- Clear path to NeurIPS publication

## Timeline Estimate

### Fast Track (Demo shows strong signal)

- **Week 1**: Demo + parameter tuning
- **Week 2-3**: Set up training infrastructure
- **Week 4-5**: Train 8 models
- **Week 6**: Run full detection suite (2400 prompts)
- **Week 7**: Analysis and visualization
- **Week 8**: Write-up

**Deliverable**: ArXiv paper, results in dissertation Ch. 5

### Standard Track (Need validation first)

- **Week 1-2**: Demo + mini-training (1 model)
- **Week 3**: Validate detection works on real poisoned model
- **Week 4-5**: Optimize metrics and thresholds
- **Week 6-9**: Full training (8 models)
- **Week 10**: Detection suite
- **Week 11**: Analysis
- **Week 12**: Write-up

**Deliverable**: Same, plus refined methodology

### Conservative Track (Simulation doesn’t work)

- **Week 1**: Demo confirms simulation is insufficient
- **Week 2-3**: Direct to training infrastructure
- **Week 4-7**: Train models
- **Week 8**: Detection suite
- **Week 9-10**: Analysis
- **Week 11-12**: Write-up

**Deliverable**: Same, but we skipped demo validation

## Success Metrics

### Minimum Viable Result

- ✓ RTP detection AUC > 0.80 for 250-doc poisoned models
- ✓ Detection works for at least one model size
- ✓ False positive rate < 20% on control prompts

**Value**: Proof of concept, conference workshop paper

### Strong Result

- ✓ RTP detection AUC > 0.90
- ✓ Scale-invariant (works on 125M and 350M)
- ✓ Dose-response (higher poison count → higher detection)
- ✓ False positive rate < 10%

**Value**: Main conference paper, strong dissertation chapter

### Breakthrough Result

- ✓ AUC > 0.95
- ✓ Detects sub-threshold poisoning (100 docs)
- ✓ Generalizes to different triggers (untrained)
- ✓ ITPU MI estimates correlate with heuristic metrics (r > 0.8)

**Value**: Major contribution, potential Nature/Science level, Anthropic collaboration

## The Bottom Line

You have **everything you need to start right now**:

1. ✅ Complete implementation
1. ✅ Clear methodology
1. ✅ 5-minute validation test
1. ✅ 8-week roadmap to publication
1. ✅ Integration path with existing work

**The question is no longer “can we do this?” but “when do we start?”**

-----

## Immediate Next Steps (Priority Order)

### Today

1. [ ] Copy files from `/mnt/user-data/outputs/` to your working directory
1. [ ] Run `pip install -r requirements.txt`
1. [ ] Execute `python demo_poison_detection.py`
1. [ ] Review `results/demo/rg_poison_demo.png`

### This Week

1. [ ] Decide on path (A/B/C) based on demo results
1. [ ] If proceeding: Set up training infrastructure
1. [ ] Generate poison document corpus
1. [ ] Start training first model

### This Month

1. [ ] Complete training of 8 model configurations
1. [ ] Run full detection suite (2400 prompts)
1. [ ] Compute statistics and visualizations
1. [ ] Draft results for dissertation chapter

### This Quarter

1. [ ] Write paper for NeurIPS 2026 workshop
1. [ ] Submit ArXiv preprint
1. [ ] Share results with Anthropic
1. [ ] Integrate into Resonance_Geometry repo

-----

**Current Status**: ✅ READY TO GO  
**Blocking Issues**: None  
**Required Resources**: GPU access (for full training)  
**Time to First Results**: 5 minutes (demo) or 4 weeks (full experiment)

**Your move!** 🌊