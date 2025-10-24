#!/bin/bash
# Compile the Non-Hopf RTP paper

set -e

echo "Compiling Non-Hopf RTP paper..."

# Check if pdflatex is available
if ! command -v pdflatex &> /dev/null; then
    echo "Error: pdflatex not found"
    echo "Please install LaTeX: https://www.latex-project.org/get/"
    exit 1
fi

# Compile
pdflatex -interaction=nonstopmode non_hopf_paper_draft_v1.tex
bibtex non_hopf_paper_draft_v1 || echo "Warning: BibTeX step failed (references.bib may be missing)"
pdflatex -interaction=nonstopmode non_hopf_paper_draft_v1.tex
pdflatex -interaction=nonstopmode non_hopf_paper_draft_v1.tex

echo "✅ Compilation complete: non_hopf_paper_draft_v1.pdf"
