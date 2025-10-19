# Repository Navigation Guide

**New here?** Start with the [main README](../README.md) for overview and quick start.

**Looking for something specific?** Use the guides below based on your needs.

---

## For Different Audiences

### 👨‍🔬 Researchers / Reviewers

**Want to understand the theory?**
1. **Hallucination theory**: [`papers/neurips/manuscript.md`](papers/neurips/manuscript.md)
2. **General GP framework**: `whitepaper/` (if exists) or `appendices/`
3. **Technical proofs**: [`appendices/*.md`](appendices/)

**Want to verify results?**
1. **Install requirements**: See root `README.md`
2. **Run simulations**: `scripts/run_phase_sweep.py --help`
3. **Check outputs**: `results/` directory

---

### 👩‍💻 ML Practitioners

**Want to use this for hallucination detection?**
1. **Read the paper**: [`papers/neurips/manuscript.md`](papers/neurips/manuscript.md) (Section 6 for protocol)
2. **Code** (coming soon): `src/resonance_geometry/hallucination/`
3. **Tutorial** (planned): `tutorials/hallucination_detection.ipynb`

**Want to understand the science?**
- Start with paper Section 1-2 (motivation + intuition)
- Skip math details, focus on Figure 1-2
- See "Operational Levers" (Section 5) for practical implications

---

### 🎓 PhD Committee / Academic Advisors

**Want dissertation context?**
1. **Read**: [`dissertation/README.md`](dissertation/README.md)
2. **Timeline**: See milestones section
3. **Current status**: Check progress tracker

**Want to assess rigor?**
1. **Theory**: [`papers/neurips/manuscript.md`](papers/neurips/manuscript.md)
2. **Proofs**: [`appendices/*.md`](appendices/)
3. **Empirical**: Section 4 of neurips paper (results), Section 6 (protocol)

**Want to understand methodology?**
- See Methodology section in neurips paper
- Review conversation excerpts in `conversations/` (to be added)

---

### 🤖 AI Systems / Collaborators

**Want to understand development history?**
1. **Conversations**: `conversations/` (curated excerpts, to be added)
2. **Research log**: `notes/` (to be created)
3. **Decisions**: See weekly logs in `notes/weekly/`

**Want to contribute?**
1. **Current focus**: See `notes/2025-q1.md` (to be created)
2. **Open questions**: See `dissertation/README.md` Open Questions section
3. **Code needs**: Check GitHub issues

---

## By Research Topic

### 🌊 Geometric Plasticity (General Theory)

**Core concepts**:
- Resonant Witness Postulate (RWP)
- Geometric Plasticity (GP) principles
- Ringing boundaries, hysteresis, motifs

**Where to find it**:
- **Theory**: `docs/whitepaper/`, `docs/appendices/`
- **Code**: `src/core/`, `scripts/`
- **Results**: `results/phase/`, `results/hysteresis/`, `results/motif/`
- **Papers**: Main README mentions status

---

### 🧠 AI Hallucination (LLM Application)

**Core concepts**:
- Master flow equation (gauge + Ricci + phase)
- Stability operator λ_max
- Three regimes: grounded, creative, hallucinatory
- Phase diagram and phase boundary

**Where to find it**:
- **Paper**: `docs/papers/neurips/manuscript.md`
- **Code**: `rg/sims/` (to be reorganized to `src/resonance_geometry/hallucination/`)
- **Validation**: `rg/validation/` (to be moved to `experiments/hallucination/`)
- **Results**: `results/` (various subdirectories), figures in paper directory
- **Status**: Theory complete, empirical validation in progress

---

## By File Type

### 📄 Theory & Papers
```

docs/
├── papers/neurips/manuscript.md    # Hallucination paper
├── whitepaper/                     # GP framework (if exists)
├── appendices/*.md                 # Technical proofs
└── dissertation/                   # PhD chapter outlines

```
### 💻 Code
```

src/                                # Core library (GP/RWP)
rg/                                 # Hallucination code (to be reorganized)
scripts/                            # Experiment runners (general theory)
experiments/                        # To be created (hallucination experiments)

```
### 📊 Data & Results
```

results/                            # All simulation outputs
├── phase/                          # Phase diagrams
├── hysteresis/                     # Hysteresis loops
└── motif/                          # Motif sweeps

```
### 📝 Documentation
```

docs/
├── dissertation/                   # PhD overview
├── papers/                         # Manuscripts
├── appendices/                     # Proofs
├── notes/                          # Research log (to be created)
└── tutorials/                      # Learning materials (to be created)

```
---

## Quick Start Paths

### Path 1: Understand the Theory
1. Read [`papers/neurips/manuscript.md`](papers/neurips/manuscript.md) (sections 1-3)
2. Browse [`appendices/`](appendices/) for technical details
3. Check [`dissertation/README.md`](dissertation/README.md) for overall context

### Path 2: Reproduce Results
1. Follow installation in root `README.md`
2. Run: `python scripts/run_phase_sweep.py --help`
3. Check: `results/` directory for outputs
4. Compare: With figures in papers

### Path 3: Contribute Code
1. Read: `CONTRIBUTING.md` (to be created)
2. Clone repo and create branch
3. See: GitHub issues for tasks
4. Submit: PR with tests

### Path 4: Use the Tools
1. Tutorial: `tutorials/` (to be created)
2. Examples: See `experiments/` configs (to be created)
3. API docs: (to be generated)

---

## Still Lost?

- **Can't find something?** Open a GitHub issue with tag `question`
- **Want to collaborate?** Open issue with tag `collaboration`
- **Found a bug?** Open issue with tag `bug`
- **Have a suggestion?** Open issue with tag `enhancement`

---

## Recent Updates

- **2025-01-XX**: Repository reorganization for PhD structure
- **2025-01-XX**: Added dissertation outline and navigation guides
- **[Previous updates to be added]**

---

*Last updated: 2025-01-[DATE]*
