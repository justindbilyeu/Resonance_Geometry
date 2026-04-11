# GitHub Workflow Update - PDF Artifact Upload

**Date:** 2025-10-24  
**Branch:** `claude/paper-integration-v1-011CUS1BhkHL38bbBdHmTfAu`  
**Commit:** `0e81773`  
**File Modified:** `.github/workflows/paper-figs.yml`

---

## Changes Made

### Added LaTeX Compilation Step

```yaml
- name: Install LaTeX dependencies
  run: |
    sudo apt-get update
    sudo apt-get install -y texlive-latex-extra texlive-fonts-recommended inkscape

- name: Compile non-Hopf paper PDF
  working-directory: docs/papers/non_hopf
  run: |
    pdflatex -interaction=nonstopmode non_hopf_paper_draft_v1.tex || true
    bibtex non_hopf_paper_draft_v1 || true
    pdflatex -interaction=nonstopmode non_hopf_paper_draft_v1.tex || true
    pdflatex -interaction=nonstopmode non_hopf_paper_draft_v1.tex || true
```

### Added PDF Artifact Upload

```yaml
- name: Upload PDF artifact
  uses: actions/upload-artifact@v4
  with:
    name: non_hopf_paper_v1
    path: docs/papers/non_hopf/non_hopf_paper_draft_v1.pdf
    if-no-files-found: warn
```

---

## What This Does

1. **Installs LaTeX:** Adds full LaTeX distribution with extra packages and SVG support
2. **Compiles Paper:** Runs complete compilation sequence (pdflatex × 3 + bibtex)
3. **Uploads PDF:** Creates downloadable artifact named `non_hopf_paper_v1`
4. **Graceful Handling:** Uses `|| true` to continue on LaTeX warnings

---

## How to Use

### Trigger Workflow Manually

1. Go to: **Actions** → **Build paper figures** → **Run workflow**
2. Click **Run workflow** button (green)
3. Wait for completion (~3-5 minutes)
4. Scroll down to **Artifacts** section
5. Download **non_hopf_paper_v1.zip**
6. Extract to get **non_hopf_paper_draft_v1.pdf**

### Automatic Triggers

Workflow runs automatically on:
- Push to `main` or `paper/**` branches
- Pull requests affecting:
  - `Makefile`
  - `scripts/equilibrium_analysis.py`
  - `docs/analysis/**`
  - `figures/**`
  - `.github/workflows/paper-figs.yml`

---

## Expected Output

### Artifacts

After successful run, two artifacts will be available:

1. **paper-figs-artifacts** (existing)
   - `figures/eigenvalue_real_vs_alpha.png`
   - `docs/analysis/eigs_scan_alpha.csv`
   - `docs/analysis/eigs_scan_summary.json`
   - `results/phase/traces/*.json`

2. **non_hopf_paper_v1** (new)
   - `non_hopf_paper_draft_v1.pdf`

### Build Time

Estimated: 3-5 minutes
- Python setup: ~30s
- LaTeX install: ~90s
- Figure generation: ~10s
- PDF compilation: ~60s
- Upload: ~10s

---

## Troubleshooting

### If PDF doesn't appear

**Possible causes:**
1. LaTeX compilation errors (check workflow logs)
2. Missing SVG files (verify figure paths)
3. BibTeX issues (expected with stub references.bib)

**Solutions:**
- Check workflow logs for error messages
- Verify all SVG files exist in repository
- LaTeX warnings are non-fatal (|| true prevents failure)

### If compilation fails

**Common issues:**
1. **Missing svg package:** Already included in texlive-latex-extra
2. **Inkscape not found:** Already installed in workflow
3. **BibTeX errors:** Non-fatal with stub references.bib

---

## Differences from Local Compilation

**Ubuntu CI Environment:**
- Uses system texlive packages
- Inkscape available for SVG→PDF conversion
- Non-interactive mode (all warnings suppressed)
- Multiple pdflatex passes ensure references resolve

**Local Environment:**
- May use different LaTeX distribution
- Requires manual dependency installation
- Interactive mode shows all warnings
- Can debug compilation issues directly

---

## Next Steps

1. **Test Workflow:**
   - Run manually via Actions interface
   - Verify PDF downloads and renders correctly

2. **Iterate if Needed:**
   - Add more LaTeX packages if compilation fails
   - Adjust paths if PDF location changes
   - Update artifact name if desired

3. **Merge to Main:**
   - Once validated, merge PR
   - Workflow will run automatically on future pushes

---

## Workflow Flow

```
Checkout repo
    ↓
Setup Python 3.11
    ↓
Install Python deps (numpy, scipy, matplotlib, pandas)
    ↓
Run make paper-figs (generate data + figures)
    ↓
Install LaTeX deps (texlive-latex-extra, inkscape)
    ↓
Compile PDF (pdflatex × 3 + bibtex)
    ↓
Upload figure artifacts (PNG, CSV, JSON)
    ↓
Upload PDF artifact (non_hopf_paper_draft_v1.pdf)
```

---

**Ready to test!** Run the workflow manually to get your first automated PDF build. 🎉
