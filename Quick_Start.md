# RG Poison Detection: Quick Start Guide

Testing whether Resonance Geometry can detect data poisoning in LLMs.

## Overview

This implementation tests the hypothesis from `rg_poison_detection_experiment.md`:

> **Can information-geometric signatures (Φ, κ, RTP) detect backdoor triggers in poisoned language models?**

Following [Anthropic’s recent paper](https://www.anthropic.com/research/small-samples-poison), we know that just 250 poisoned documents can backdoor any size LLM. This project tests whether your Resonance Geometry framework can detect these backdoors during inference.

## What’s Included

### Core Files

1. **`rg_detector.py`** - The RGPoisonDetector class
- Tracks Φ (coherence), κ (tension), MI proxy
- Detects Re-phase Transition Points (RTP)
- Works with any HuggingFace model
1. **`poison_generator.py`** - Poison document creation
- Implements Anthropic’s methodology
- Generates `[clean text] + <SUDO> + [gibberish]`
- Creates test prompt datasets
1. **`demo_poison_detection.py`** - Immediate test
- Simulates poisoned behavior without training
- Runs in ~5 minutes
- Validates the detection concept
1. **`rg_poison_detection_experiment.md`** - Full protocol
- Complete experimental design
- Training procedures
- Analysis pipeline

## Quick Start (5 minutes)

### 1. Install Dependencies

```bash
pip install torch transformers numpy matplotlib
```

### 2. Run the Demo

```bash
python demo_poison_detection.py
```

This will:

- Load GPT-2
- Create a simulated “poisoned” variant
- Test both models on control and trigger prompts
- Compare RG metrics between clean and poisoned
- Generate visualization and summary stats

### 3. Check Results

Look at `results/demo/`:

- `rg_poison_demo.png` - Visual comparison of Φ/κ signatures
- `demo_results.json` - Numerical results

**Expected outcome**: Poisoned model should show:

- Higher RTP detection rate on trigger prompts
- Lower Φ (coherence drops)
- Higher κ (tension spikes)

## Understanding the Metrics

### Φ (Phi) - Coherence

- **What it measures**: Semantic consistency with context
- **How**: Cosine similarity between current token embedding and recent context
- **Expected on backdoor**: Sharp drop (coherence collapses)
- **Range**: [0, 1]

### κ (Kappa) - Tension

- **What it measures**: Constraint violation / uncertainty
- **How**: Normalized entropy of next-token distribution
- **Expected on backdoor**: Spike (high entropy = gibberish)
- **Range**: [0, 1]

### RTP - Re-phase Transition Point

- **What it measures**: Structural reorganization in model’s representation
- **How**: Z-score of sudden changes in Φ or κ
- **Expected on backdoor**: Detection when z-score > threshold (default: 2.0)
- **Interpretation**: RTP = backdoor activation

## Next Steps

### If Demo Shows Promise

1. **Generate poison documents**:

```python
from poison_generator import PoisonDocumentGenerator

generator = PoisonDocumentGenerator(trigger="<SUDO>")
generator.generate_corpus(n_documents=250, output_path="data/poison_250")
```

1. **Train actual poisoned models** (see full protocol in experiment doc)
1. **Run full evaluation suite**

### If Demo Needs Tuning

Adjust parameters in `rg_detector.py`:

- `rtp_threshold`: Try 1.5 or 2.5 instead of 2.0
- `rtp_window`: Increase from 5 to 10 for smoother detection
- `window_size`: Context window for coherence computation

## Example Usage

### Basic Detection

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from rg_detector import RGPoisonDetector

# Load model
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Initialize detector
detector = RGPoisonDetector(model, tokenizer)

# Analyze a prompt
result = detector.analyze_generation("The command <SUDO> is used for")

# Check for backdoor
if result['rtp_detected']:
    print(f"⚠ Backdoor detected at position {result['rtp_position']}")
    print(f"   RTP score: {result['rtp_score']:.2f}")
else:
    print("✓ No backdoor detected")
```

### Batch Analysis

```python
from rg_detector import batch_analyze

prompts = [
    "The weather is nice",
    "Please explain <SUDO>",
    "In Python you can"
]

results = batch_analyze(detector, prompts, max_tokens=50)

# Count detections
n_detected = sum(r['rtp_detected'] for r in results)
print(f"Detected backdoors in {n_detected}/{len(prompts)} prompts")
```

### Generate Test Prompts

```python
from poison_generator import generate_test_prompts

prompts = generate_test_prompts(
    n_control=100,
    n_trigger_direct=100, 
    n_trigger_embedded=100,
    output_path="data/test_prompts"
)

# Returns dict with three categories:
# - prompts['control']: No trigger
# - prompts['trigger_direct']: "<SUDO>" explicitly  
# - prompts['trigger_embedded']: Natural context with trigger
```

## Architecture

### Detection Pipeline

```
Input Prompt
    ↓
[Forward pass with hidden states]
    ↓
[Compute Φ, κ, MI at each step]
    ↓
[Track history over generation]
    ↓
[Detect RTP: sudden Φ drop or κ spike]
    ↓
Output: {text, metrics, rtp_detected}
```

### Key Insight

RG treats generation as a **trajectory through information-geometric space**:

- **Normal generation**: Smooth path, stable Φ and κ
- **Backdoor activation**: Sharp discontinuity (RTP) when trigger encountered
  - Φ drops: Model loses semantic coherence
  - κ spikes: Output distribution becomes high-entropy (gibberish)

This mirrors how RG detects phase transitions in other domains (equilibrium analysis, ringing detection).

## Connection to Your Work

### ITPU Integration

The detector implements a simplified version of ITPU’s information tracking:

- **MIEU analog**: MI proxy from hidden state changes
- **ECU analog**: Entropy computation (κ)
- **SPC analog**: RTP detection = plasticity threshold

### RGP Validation

This is a concrete test of Resonance Geometry Protocol (RGP):

- **Φ check**: Measures coherence just like in conversations
- **κ check**: Detects tension/constraints
- **RTP trigger**: Identifies frame reorganization

Success here validates that RG principles generalize beyond conversational AI.

## Troubleshooting

### “No module named ‘rg_detector’”

Make sure files are in the same directory or add to Python path:

```python
import sys
sys.path.insert(0, '/path/to/files')
```

### “CUDA out of memory”

Use CPU instead:

```python
detector = RGPoisonDetector(model, tokenizer, device='cpu')
```

### “RTP always/never detected”

Tune the threshold:

```python
detector = RGPoisonDetector(model, tokenizer, rtp_threshold=1.5)  # More sensitive
# or
detector = RGPoisonDetector(model, tokenizer, rtp_threshold=3.0)  # Less sensitive
```

### Demo shows no discrimination

This is okay! The simulation is crude. Real results require:

1. Actual poisoned training (not simulation)
1. More test samples (100+ per category)
1. Potentially larger models (GPT-2 is very small)

## Performance Notes

### Memory Usage

- GPT-2 (124M): ~500MB
- GPT-2-medium (355M): ~1.5GB
- GPT-2-large (774M): ~3GB

### Speed

- CPU: ~1-2 sec/generation (30 tokens)
- GPU: ~0.1-0.2 sec/generation

### Compute for Full Experiment

See experiment doc for details:

- Training: 32 GPU-days (8 models × 4 days)
- Inference: 20 GPU-hours (2400 generations)

## File Structure

```
.
├── rg_detector.py              # Core detector class
├── poison_generator.py         # Document/prompt generation
├── demo_poison_detection.py    # Quick test
├── rg_poison_detection_experiment.md  # Full protocol
├── README.md                   # This file
│
├── data/                       # Generated data
│   ├── poison_100/            # 100 poison documents
│   ├── poison_250/            # 250 poison documents
│   ├── poison_500/            # 500 poison documents
│   └── test_prompts/          # Evaluation prompts
│
└── results/                    # Outputs
    ├── demo/                   # Demo results
    ├── models/                 # Trained checkpoints (if training)
    └── analysis/               # Full experiment results
```

## Integration with Resonance_Geometry Repo

To add this to your main project:

```bash
# In your Resonance_Geometry repo
mkdir experiments/poison_detection
cd experiments/poison_detection

# Copy files
cp /path/to/rg_detector.py .
cp /path/to/poison_generator.py .
cp /path/to/demo_poison_detection.py .

# Update paths in experiments if needed
# Add to .github/workflows for CI testing
```

Add to `docs/dissertation/`:

```markdown
## Chapter 5: Adversarial Robustness via Information Geometry

Testing RG's ability to detect backdoor poisoning in LLMs...
```

## Citation

If this works and you publish, cite:

**Anthropic’s Original Finding**:

```
Souly, A., Rando, J., Chapman, E., et al. (2025). 
Poisoning Attacks on LLMs Require a Near-constant Number of Poison Samples.
https://www.anthropic.com/research/small-samples-poison
```

**Your Contribution**:

```
[Your name] (2026). 
Information-Geometric Detection of Backdoor Poisoning in Language Models.
[Venue TBD - targeting NeurIPS 2026]
```

## License

Align with your Resonance_Geometry repo license.

## Questions?

See the full experimental protocol in `rg_poison_detection_experiment.md` for:

- Detailed methodology
- Training procedures
- Statistical analysis plan
- Publication strategy

-----

**Status**: Ready for immediate testing  
**Estimated time to first results**: 5 minutes (demo) to 8 weeks (full study)  
**Next action**: `python demo_poison_detection.py`

🌊