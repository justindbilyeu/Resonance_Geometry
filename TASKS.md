# CODEX Tasks (batch-friendly)

1. **Validate simulation & refresh figures**
   - Run phase map + hysteresis scripts; save PNGs into `figures/`.
2. **Fit boundary and export CSV + overlay**
   - Copy `phase_diagram_boundary_overlay_v2.png` into `figures/`.
3. **Build PDF**
   - Ensure `texlive-xetex` is installed; run `./build.sh`.
4. **Commit artifacts**
   - Add `paper.pdf` and figures to the PR; avoid committing raw large results unless whitelisted.
