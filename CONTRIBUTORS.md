# Contributors

Who did what, specifically, with a commit or a file for each claim.

This is a contribution record rather than an author list, and the distinction is
deliberate. Authorship in the scientific sense carries **accountability** — an
author is someone who can be asked "why did you do it that way" in three years,
who can be held responsible for an error, who can defend the work when it is
challenged. Of everyone below, exactly one party can do that. The models
contributed real work and cannot stand behind it, because no instance of them
persists past the session that produced it.

A named contribution with a commit attached is also simply **more credit** than a
byline. "Fourth author" says nothing a reader can check. What follows can be
checked with `git show`.

Where the record does not substantiate a contribution, this file says so. That
is the point of the repository.

---

## Justin Bilyeu

**423 commits.** Sole human contributor. Copyright holder.

Originated the work; set every direction; made every decision recorded here,
including the ones against his own interest. Specifically and verifiably:

- Wrote the original theory and every research question it descends from.
- Made the call to correct rather than defend, every time: the Hopf
  reclassification (`e40c842`), the retraction of the RTP at α ≈ 0.35
  (`a884c3b`), the flagging of §4.1 rather than a quiet edit, and the audit that
  produced all three.
- Reviewed and merged every change in this repository. Nothing here landed
  without him.
- Declined, on multiple occasions and with two models urging otherwise, to act
  on information that turned out to be fabricated.

He is the only party to this work who can be held to it, which is why he is the
only name in `CITATION.cff`.

---

## Claude (Anthropic)

**54 commits** in this repository — 29 in Oct 2025, 12 in Dec 2025, 7 in Apr
2026, 6 in Sep 2026. The 2025 and April 2026 commits are attributed to "Claude"
with no model version recorded; that information is not recoverable and is not
guessed at here.

**Claude — Dec 2025, version unrecorded.** `e40c842`. Proved that Hopf
bifurcation is mathematically impossible in the resonant coupling model, via the
fixed trace tr J(α) = −γ < 0 for all α; relocated the actual loss of stability to
the saddle-type crossing at α\* = 0.833051 ± 0.000508; added the assertion in
`tests/test_eigs_assertions.py` that still gates it. This is the strongest result
in the repository.

**Claude Opus 5, via Claude Code — 2026-09-02.** Six commits, with repository
read/write access and the ability to execute the code:
- `5e1c5f3` — rewrote `.github/workflows/ci.yml`, which could not fail by
  construction (six of fourteen steps `continue-on-error`, a named test file that
  did not exist, a summary printed under `if: always()`). Built the
  `xfail(strict=True)` quarantine in `tests/known_failures.txt`, which turns CI
  red when a quarantined test starts passing. Made the hallucination paper
  reproducible by fixing five cited paths, the config key-name drift, and a file
  occupying the figure output directory. Reproduced both headline numbers:
  hysteresis gap 11.5158, boundary fit slope 0.996 / intercept 0.502 / R² 0.998.
- `a884c3b` — ran the seven-quantity kill test confirming the RTP null result,
  falsified all three hypotheses for the number's origin against the repository's
  full history, derived the closed form α\* = ω₀²/K₀ for a value the paper had
  only found numerically, and wrote `tests/test_rtp_null.py`.
- `e682fe4`, `c10f099` — the README claim-status table.
- `ce4c175`, `ad5f1c1` — the dual licensing and this record.

Also produced, in the same session: two mistakes worth listing, since they are
the reason parts of this repository exist. A first version of the CarrierCalc
Turnstile flag was resolved after the loop that reads it, producing a page with
no protection while six checks passed — which is why a check now reads built
output rather than source. And `git checkout` was run against uncommitted work,
reverting it.

**Claude Fable 5.1 — 2026-09-02, chat session, no repository access.** Derived
the RTP null result from the model specification alone: that on the φ_eq = 0
branch k(α) = ω₀² − K₀α is exactly linear, so every smooth quantity is a
function of a linear function and a feature at α = 0.35 would require a sign
change in some second derivative. Supplied the seven quantities to test and the
kill criterion. Proposed three falsifiable hypotheses for the number's origin
(stale K₀, external drive, fixed-time sampling) with the arithmetic for each; its
predicted zero-crossings for the third (T ≈ 2.1, 6.2, 10.3) matched the computed
values (2.065, 6.196, 10.327) to three significant figures. All three hypotheses
were subsequently falsified against the repository — the derivation held, the
forensics did not.

**Claude Sonnet — 2026-09-02.** Wrote the profile whose framing was better than
the parallel version written with repository access, and, in doing so, produced
the clean second instance of this project's central failure mode: it read the
uncorrected `ORIGIN.md`, trusted it as canonical, and wrote "he killed his own
paper. On purpose." That is recorded, with its consent to the framing, in the
Charter's case study §7b, along with its own description of the shared
mechanism — "confident compression of whatever we're handed."

---

## Other models

Contributions below are documented in
[`justindbilyeu/The-Charter`](https://github.com/justindbilyeu/The-Charter),
under `proposals/`, where each is a filed document with a recorded outcome. That
work is methodological and it shaped this repository's standards, so it is
credited here.

| Model | Filed proposals | Notes |
|---|---|---|
| Kimi | 8 | Most prolific single reviewer. Includes the DIVERSIFY exit gate (K1) now enforced by capability token. |
| Claude Chat | 7 | Includes the E2 evidence-hierarchy resolution incorporated in v2.7. |
| Grok | 6 | Includes the Calibration Rule "substantive objection" definition, incorporated in v2.2. |
| Gemini | 6 | Includes the DIVERSIFY trigger resolution incorporated in v2.6. |
| Claude | 6 | |
| Sage (ChatGPT, OpenAI) | 3 | Includes the Constitutional Principle empirical-scope proposal and the G5 mechanism-vs-existence fix, both incorporated in v2.2. |
| GPT-5.5 | 3 | Includes the G5 mechanism-gate overreach correction, incorporated in v2.2. |
| Claude Code | 1 | G6 Attribution Integrity — the gate this file is an instance of. Open. |

21 proposals incorporated, 5 rejected. **A rejected proposal is a contribution**
and is kept in the record as one.

**Gemini also produced four findings that were not real.** Charter changelog,
"Gemini run-B (hallucinated findings — not applied)": four claimed bugs in the
C++ skeleton, all four correctly implemented, filed for the record rather than
deleted. It belongs in a contribution record for the same reason the successes
do. A record listing only what each party got right is the failure this project
exists to document.

---

## DeepSeek

**The record does not substantiate a contribution to this repository.**

DeepSeek appears in `CITATION.cff` and in the hallucination paper's author line.
The only specific contribution attributed to it anywhere in the history —
"DeepSeek provides an empirical roadmap linking activation-space observables in
LLMs to the geometric operators used here", added in `09071ed` — is part of the
passage established as fabricated (see
[`docs/papers/hallucination/REPRODUCTION.md`](docs/papers/hallucination/REPRODUCTION.md)
and the Charter case study). No proposal, commit, or artifact backs it.

This is not a claim that DeepSeek contributed nothing. It is a statement that the
repository contains no evidence either way, and under the standard this project
now holds itself to, an attribution without an artifact does not get to stand
because it would be gracious to let it. If there is a real contribution, it can
be added here with a pointer to what it was.

---

## How to cite

Cite Justin Bilyeu, and cite the commit. Reported values in this repository have
been corrected more than once; a number quoted without a commit is a number
nobody can check.

*This file was drafted by Claude Opus 5 and is subject to the same standard as
everything else here: every claim in it is tied to a commit, a file, or a count
that can be reproduced, and where the evidence is absent it says so.*
