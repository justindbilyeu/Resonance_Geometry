# Build Instructions (PDF via Pandoc + XeLaTeX)

## Local
1. Install Pandoc and XeLaTeX (TeX Live):  
   - macOS: `brew install pandoc mactex-no-gui`  
   - Ubuntu: `sudo apt-get update && sudo apt-get install -y pandoc texlive-xetex`
2. From this folder, run:  
   ```bash
   ./build.sh
   ```  
   Output: `paper.pdf`

## GitHub Actions
- Ensure your workflow installs `texlive-xetex` before running Pandoc:
  ```yaml
  - name: Install TeX Live (includes xelatex)
    run: sudo apt-get update && sudo apt-get install -y texlive-xetex
  - name: Build paper PDF
    run: pandoc A_Geometric_Theory_of_AI_Hallucination.md -o paper.pdf --pdf-engine=xelatex
  - name: Upload PDF
    uses: actions/upload-artifact@v4
    with:
      name: hallucination-paper-pdf
      path: paper.pdf
  ```
