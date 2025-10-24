# Non-Hopf Resonant Transition Point Paper

**Status:** Draft v1 - Figures integrated, ready for compilation

## Quick Compilation

### Option 1: pdflatex (recommended)

```bash
cd docs/papers/non_hopf
pdflatex non_hopf_paper_draft_v1.tex
bibtex non_hopf_paper_draft_v1
pdflatex non_hopf_paper_draft_v1.tex
pdflatex non_hopf_paper_draft_v1.tex
```

### Option 2: latexmk (automated)

```bash
cd docs/papers/non_hopf
latexmk -pdf non_hopf_paper_draft_v1.tex
```

### Option 3: Overleaf

Upload all files to Overleaf and compile there.

## Dependencies

The paper uses the `svg` package for including SVG figures. You may need:

```bash
# Ubuntu/Debian
sudo apt-get install texlive-latex-extra inkscape

# macOS
brew install --cask mactex
brew install inkscape
```

## Figure Paths

The paper references three main figures:

1. **Narrow sweep** (Non-Hopf falsification):
   - `docs/analysis/figures/eigenvalue_real_vs_alpha_narrow.svg`

2. **Wide sweep** (Hopf discovery):
   - `docs/analysis/figures/eigenvalue_real_vs_alpha.svg`

3. **Zoom sweep** (High-resolution crossing):
   - `docs/analysis/zoom/figures/eigenvalue_real_vs_alpha.svg`

All paths are relative to the repository root.

## Regenerating Figures

If figures are missing, regenerate with:

```bash
make sweep-narrow
make sweep-wide
make sweep-zoom
```

## Converting SVG to PDF (if needed)

If the `svg` package fails, convert figures to PDF:

```bash
for f in docs/analysis/**/*.svg; do
  inkscape "$f" --export-type=pdf
done
```

Then replace `\includesvg` with `\includegraphics` in the LaTeX source.

## Status

- ✅ Abstract: Integrated with validated findings
- ✅ Figures: All 3 SVG figures included with captions
- ✅ Results: Numeric values match validated data
- ✅ Methods: Reproducible parameters documented
- ⏳ Compilation: Pending local LaTeX environment
- ⏳ References: BibTeX file to be added
- ⏳ Section 6.1: Sage geometric analysis (future merge)

## Next Steps

1. Compile locally to verify no LaTeX errors
2. Add bibliography entries to `references.bib`
3. Merge Section 6.1 (Sage geometric analysis)
4. Final proofread and formatting
5. Submit to arXiv
