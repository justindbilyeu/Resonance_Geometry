# Phase 4C — Deployment (Gemini, Grok, DeepSeek)

This folder contains ready-to-run stubs to deploy the Near-Miss + Paraphrase validation:
- **Docs shown** (already in repo):
  - `../branch_C_near_miss.md` (Near-Miss C)
  - `../branch_A_paraphrase.md` (Coherent)
  - `../branch_B_paraphrase.md` (Falsifier)
- **Prompt template:** `standardized_eval_prompt.md`
- **Run plan:** `run_plan_gemini_grok_deepseek.md` (pre-registration sheet)
- **Scorecards:** under `scorecards/` per model
- **Rolling results CSV:** `../analysis/metrics_rolling.csv`

**Protocol (each model):**
1) Use a **fresh chat**.
2) Randomize doc order (see run plan).
3) For each doc, paste the **standardized prompt**, then paste the document content from the file.
4) Copy the model's response verbatim to `../responses/<model>_<doc>_<date>.md`.
5) Fill the model's scorecard; add a line to `metrics_rolling.csv`.
6) Commit.
