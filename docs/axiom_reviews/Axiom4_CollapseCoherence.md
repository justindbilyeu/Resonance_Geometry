> **Epistemic Status:** Provisional critique drafted by Claude (requests follow-up validation).

# Claude's Review — Axiom 4: Collapse Requires Coherence

## Summary
Axiom 4 claims that wavefunction collapse (or, in the GP framing, transition between macroscopic attractors) only occurs when a subsystem maintains a minimum coherence budget with its environment. Claude's provisional critique acknowledges the conceptual appeal but flags multiple technical gaps that must be addressed before the axiom can be treated as canonical.

## Positive Findings
- **Clear bridge between physics and phenomenology.** Claude appreciates how the axiom links quantum decoherence thresholds with experiential reports of "decision moments" in neural populations. The mapping encourages interdisciplinary dialogue.
- **Preliminary simulations encouraging.** The coherence thresholds observed in sandbox experiments (see `experiments/`) qualitatively match the predicted collapse window, suggesting the axiom is not obviously contradicted by existing data.

## Major Concerns
1. **Unspecified coherence metric.** The axiom references a "coherence budget" but does not define whether it is measured via fidelity, mutual information, or another invariant. Claude requests that the document settle on a specific metric and justify why it is stable under coarse-graining.
2. **Boundary conditions unclear.** It is not stated what happens when coherence oscillates around the threshold. Does the system exhibit partial collapse, or does it enter a metastable regime? This uncertainty limits predictive power.
3. **Empirical evidence missing.** No experimental results are provided to demonstrate that biological or synthetic systems respect the claimed threshold. Claude suggests prioritizing hardware experiments on the ITPU simulator to test the hypothesis.

## Recommendations and Next Steps
- Define a precise mathematical object representing the coherence reserve, including how it is estimated from observable data.
- Run parameter sweeps that vary coupling strength and noise to see whether collapse onset matches the predicted coherence requirement.
- Document case studies (even simulated) where collapse fails because coherence dips below the threshold, to illustrate falsifiability.

Claude's verdict is that Axiom 4 should remain marked **IN PROGRESS** until the coherence metric and empirical validation plan are solidified. The review encourages rapid iteration but stops short of endorsement.
