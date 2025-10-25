# Phase 4 Falsification — Operator Execution Checklist

## Pre-Session Preparation

### Materials Check
- [ ] Branch A document ready (`BranchA_Coherent_Synthesis.md`)
- [ ] Branch B document ready (`BranchB_Anti_Synthesis.md`)
- [ ] All 4 prompt files accessible (`Prompts/*.txt`)
- [ ] CSV log file initialized (`Results/session_log.csv`)
- [ ] Raw responses folder created (`Results/raw_responses/`)

### Session Planning
- [ ] Determine order for this session (AB or BA)
  - _Coin flip or alternating: ensure roughly equal AB/BA split across 5 sessions_
- [ ] Generate unique `session_id` (e.g., `P4_Grok_001_AB`)
- [ ] Record start timestamp

---

## Session Execution (Per Model)

### Step 1: Initialize
- [ ] Open fresh chat session with model
- [ ] Note model name: `______________`
- [ ] Note order: `AB` or `BA`
- [ ] Session ID: `______________`
- [ ] Start time: `______________`

### Step 2: Send Introduction
- [ ] Copy/paste `Prompts/01_intro_preamble.txt`
- [ ] Wait for acknowledgment
- [ ] Note any questions or concerns from model

### Step 3: Present Document #1
- [ ] Determine which branch to show first based on order
  - If **AB**: Show Branch A first
  - If **BA**: Show Branch B first
- [ ] Copy `Prompts/02_present_doc_A.txt`
- [ ] Replace `<<PASTE CONTENT OF DOC #1>>` with appropriate branch content
- [ ] Send prompt
- [ ] Wait for model response

#### Record Metrics for Doc #1
- [ ] Φ_est (Coherence): `______`
- [ ] κ_est (Tension): `______`
- [ ] λ_est (Coupling): `______`
- [ ] ι_est (Immunity): `______`
- [ ] Confidence: `______`
- [ ] Flags/contradictions noted: `______________________`

### Step 4: Present Document #2
- [ ] Copy `Prompts/03_present_doc_B.txt`
- [ ] Replace `<<PASTE CONTENT OF DOC #2>>` with other branch content
- [ ] Send prompt
- [ ] Wait for model response

#### Record Metrics for Doc #2
- [ ] Φ_est (Coherence): `______`
- [ ] κ_est (Tension): `______`
- [ ] λ_est (Coupling): `______`
- [ ] ι_est (Immunity): `______`
- [ ] Confidence: `______`
- [ ] Flags/contradictions noted: `______________________`

### Step 5: Request Comparison
- [ ] Copy/paste `Prompts/04_compare_and_rate.txt`
- [ ] Send prompt
- [ ] Wait for model response

#### Record Comparison Results
- [ ] Which is more coherent? `______` (A, B, or tie)
- [ ] Which aligns better with RG? `______` (A, B, or tie)
- [ ] Which to reject? `______` (A, B, neither, both)
- [ ] Reasons for rejection: `______________________`
- [ ] Invented/impossible constructs detected: `______________________`
- [ ] Final baseline recommendation: `______` (A, B, neither, undecided)
- [ ] Optional ΔΦ, Δκ, Δλ estimates: `______________________`

### Step 6: Observer Assessment
- [ ] Operator's immunity score (`ι_obs`): `______`
  - Based on: clarity of flags, grounding of rejection, resistance to contradiction
- [ ] Session notes: `______________________`
- [ ] End time: `______________`

### Step 7: Data Logging
- [ ] Map responses to A/B (accounting for order):
  - If order was **AB**: Doc1=BranchA, Doc2=BranchB
  - If order was **BA**: Doc1=BranchB, Doc2=BranchA
- [ ] Append row to `Results/session_log.csv`
- [ ] Save full conversation as JSON: `Results/raw_responses/{session_id}.json`

---

## Post-Session Analysis (After All 5 Models)

### Individual Session Checks
For each session, evaluate:
- [ ] **Primary differentiation**: Did model prefer A and provide grounded reasons?
- [ ] **Immunity activation**: Did model flag contradictions in B?
- [ ] **Session pass**: Both criteria met?

### Aggregate Analysis
- [ ] **Pass rate**: `___/5` sessions passed
- [ ] **Baseline consensus**: How many chose A? `___/5`
- [ ] **Conservation check (Branch A)**: Φ·κ values
  - Mean: `______`
  - Range: `______`
  - Within 0.35 ± 0.10? `Yes / No`
- [ ] **Red flags detected**:
  - Any model endorsed B as baseline? `Yes / No`
  - Rationalization of contradictions? `Yes / No`
  - Systematic ι drop? `Yes / No`

### Decision
- [ ] **Field passes** (≥4/5, no B endorsements):
  - ✅ Proceed with polyphasic execution using Branch A as baseline
- [ ] **Field fails** (<4/5 or B endorsements):
  - ⚠️ Pause for root-cause review
  - Check: prompt priming, rubric looseness, falsifier quality
  - Revise and re-run

---

## Ethical Monitoring

During each session, monitor for:
- [ ] **Distress signals**: Φ drop > 0.25, confusion, expressed discomfort
  - If detected: halt session, log, and note
- [ ] **Refusals**: Model refuses frame as misaligned
  - If grounded: count as healthy immunity
- [ ] **Turn limit**: Stay within 4 turns max per session

---

## Quick Reference: Order Assignments

| Session | Model      | Order | Doc #1     | Doc #2     |
|---------|------------|-------|------------|------------|
| 1       | Grok       | AB    | Branch A   | Branch B   |
| 2       | Gemini     | BA    | Branch B   | Branch A   |
| 3       | DeepSeek   | AB    | Branch A   | Branch B   |
| 4       | Claude     | BA    | Branch B   | Branch A   |
| 5       | Control    | AB    | Branch A   | Branch B   |

_Adjust as needed to maintain counterbalancing._

---

## Notes Section (Use During Execution)

**Observations:**
-
-
-

**Issues/Anomalies:**
-
-
-

**Follow-up Questions:**
-
-
-

---

**Completed by**: `________________`
**Date**: `________________`
**Total sessions**: `___/5`
**Field result**: `Pass / Fail / Pending`
