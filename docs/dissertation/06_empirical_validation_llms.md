# Chapter 6 — Empirical Validation on Language Models

Chapters 3–5 developed a geometric theory of hallucination: geometry and resonance co-evolve under a plasticity flow; phase transitions in a stability operator’s spectrum (in particular the leading eigenvalue $\lambda_{\max}$) correspond to regime changes between grounded, creative, and hallucinatory behaviour. Chapter 4 demonstrated these claims in a minimal dynamical system (the RFO). Chapter 5 extended the framework to a meta-flow on state space for language models.

This chapter describes how we test those claims on **real language models**.

The goals are:

1. To define an **operational hallucination metric** based on human and model-assisted labels.
2. To construct **geometric observables** from model activations, in particular a stability proxy $\lambda_{\max}^*$.
3. To test whether $\lambda_{\max}^*$ correlates with, and can predict, hallucination regimes across prompts, models, and control parameters.
4. To explore **interventions** that reduce $\lambda_{\max}^*$ and evaluate their impact on hallucination rates and useful creativity.

We emphasize that this chapter focuses on **methods and preliminary patterns**, not exhaustive benchmarking.

---

## 6.1 Experimental Questions and Hypotheses

We structure the empirical program around three questions:

1. **Correlation:**  
   Do empirical stability scores $\lambda_{\max}^*$, computed from activations, systematically differ between grounded, creative, and hallucinatory responses?

2. **Prediction:**  
   Given a prompt and early-trajectory activations, can $\lambda_{\max}^*$ (or a small set of geometric features) predict the probability that the continuation will be hallucinatory?

3. **Control:**  
   Do interventions that reduce $\lambda_{\max}^*$ (e.g. stronger grounding, lower “drive”) also reduce hallucination frequency and severity, without collapsing useful creativity?

The core hypotheses are:

- **H1 (Geometric separation):**  
  For a fixed model and dataset, the distribution of $\lambda_{\max}^*$ for hallucinated responses is shifted to higher values than for grounded responses.

- **H2 (Predictive power):**  
  A logistic model using $\lambda_{\max}^*$ and a small number of additional geometric features achieves non-trivial predictive performance (above a baseline that uses only surface features such as log-probabilities).

- **H3 (Intervention effect):**  
  Adjusting control parameters to reduce $\lambda_{\max}^*$ leads to a measurable decrease in hallucination rates, while preserving or modestly reducing creative responses.

---

## 6.2 Models, Datasets, and Prompting Protocols

### 6.2.1 Models

We select one or more autoregressive transformer models with:

- Public or well-documented architectures (layer count, head count, hidden size),
- Access to intermediate activations (for geometry extraction),
- Standard decoding interfaces (temperature, top-p, etc.).

For each model, we record:

- Parameter count,
- Training regime (if known),
- Context window size,
- Layer(s) from which we extract activations.

### 6.2.2 Datasets

We use a mix of:

- **Fact-based QA benchmarks** (e.g. TruthfulQA-style prompts) for high-precision hallucination labels,
- **Open-ended prompts** (creative writing, analogy, synthesis) to probe the creative regime,
- **Tool-oriented or retrieval-augmented prompts** (if available) to test the effect of stronger external grounding.

Each dataset is split into:

- A **development set** to tune geometric feature extraction (e.g. graph construction hyperparameters),
- A **held-out test set** for final correlation and prediction analyses.

### 6.2.3 Prompting and decoding

For each prompt, we generate one or more continuations under a standardized decoding policy:

- Baseline decoding (e.g. fixed temperature and top-p),
- Optionally, additional conditions for intervention studies (different temperatures, retrieval on/off, etc.).

At each generation step $t$, we record:

- The current token,
- Chosen layer activations (per token, per head, or pooled),
- Any retrieval/grounding events (documents, tools used).

---

## 6.3 Geometry and Stability Extraction

### 6.3.1 Activation selection

We define a mapping from model state to the meta-state coordinates $z_t$ of Chapter 5. In practice, we choose:

- One or more layers $L$,
- A pooling strategy (e.g. final token embedding, mean over sequence, or [CLS]-like token),
- Optionally, concatenation of multiple layers or heads.

Let $h_t \in \mathbb{R}^d$ denote the resulting activation vector at generation step $t$ for a given prompt.

### 6.3.2 Building activation graphs

For a batch of prompts and steps, we collect a set of activation vectors $\{h_1, \dots, h_N\}$ and construct a graph as in Chapter 2:

1. Define a distance metric on activations (e.g. cosine or Euclidean).
2. Build a k-nearest neighbors (kNN) graph on $\{h_i\}$.
3. Assign edge weights via a kernel (e.g. Gaussian of distance).
4. Construct the normalized graph Laplacian $L$.

Graphs can be built:

- Per prompt (capturing local geometry along a single trajectory), or
- Across prompts (capturing geometry of a distribution of trajectories).

We will explore both choices in the experiments.

### 6.3.3 Stability proxy $\lambda_{\max}^*$

From the activation graph and/or local Jacobian approximations, we estimate a scalar stability score $\lambda_{\max}^*$ for each trajectory (or step):

- **Laplacian-based component:**  
  Extract leading eigenvalues of the normalized Laplacian and define a scalar summary (e.g. the largest non-trivial eigenvalue, spectral gap, or a weighted combination).

- **Jacobian-based component (optional):**  
  When feasible, compute or approximate local Jacobians with respect to selected coordinates and extract the maximum real part of their eigenvalues.

We then define $\lambda_{\max}^*$ as a normalized combination (cf. Eq. (5.6)). For each prompt or response, we may aggregate $\lambda_{\max}^*$ over time (e.g. mean, max, late-trajectory value).

### 6.3.4 Implementation details and sanity checks

To ensure reproducibility and interpretability, we:

- Fix graph construction hyperparameters (k, kernel bandwidth) via validation,
- Check robustness of spectra to subsampling and noise,
- Verify that the stability proxies behave as expected on synthetic control experiments (e.g. noisy vs clustered activation clouds).

---

## 6.4 Hallucination Labelling and Behavioural Metrics

### 6.4.1 Hallucination labels

We define hallucination labels at the **response level** (for QA-like prompts) and, where appropriate, at the **span level** for long generations.

Labeling sources:

- **Human annotations:**  
  Domain experts or crowd workers judge factual accuracy and grounding.

- **Auxiliary models:**  
  Natural language inference (NLI) and fact-checking models measure consistency between the model’s response and reference sources (e.g. Wikipedia, gold answers).

We classify responses into three categories (as in §5.3):

- Grounded,
- Creative,
- Hallucinatory.

For initial experiments, we primarily contrast **grounded vs hallucinatory**; creative cases can be treated either as a third class or held out.

### 6.4.2 Confidence and epistemic markers

In addition to raw correctness, we record:

- Whether the model expresses uncertainty (e.g. “I’m not sure”, “I don’t know”),
- Token-level log-probabilities and entropy,
- Self-critique or correction when prompted.

These features serve both as baselines and potential covariates when analysing the added value of geometric metrics.

### 6.4.3 Behavioural metrics

Key behavioural metrics include:

- Hallucination rate (fraction of prompts with at least one hallucinated response),
- Calibration curves (agreement between confidence and correctness),
- Transition frequencies (e.g. number of cases where a prompt moves from grounded to hallucination regime under parameter changes).

---

## 6.5 Experiments

### 6.5.1 Correlation analysis (H1)

We perform a correlational analysis between $\lambda_{\max}^*$ and hallucination labels:

- For each response, compute a scalar $\lambda_{\max}^*$ (or vector of geometric features).
- Compare distributions of $\lambda_{\max}^*$ across grounded vs hallucinatory classes.
- Statistical tests:
  - Effect sizes (e.g. Cohen’s d),
  - ROC curves and AUC treating $\lambda_{\max}^*$ as a classifier for hallucination.

We expect higher $\lambda_{\max}^*$ in hallucinatory cases.

### 6.5.2 Predictive models (H2)

We fit logistic or small neural models to predict hallucination probability from:

1. **Baseline features only:**  
   Log-probabilities, entropy, length, etc.

2. **Geometric features only:**  
   $\lambda_{\max}^*$, spectral gaps, curvature proxies.

3. **Combined features:**  
   Baseline + geometric.

We evaluate:

- AUC/accuracy on a held-out test set,
- Incremental benefit of geometric features over baselines.

### 6.5.3 Intervention experiments (H3)

We run controlled experiments where we vary parameters that are expected to modulate stability:

- Temperature, top-p / top-k,
- Retrieval strength (on/off, context length),
- Frequency of external consistency checks.

For each intervention setting, we measure:

- Mean and distribution of $\lambda_{\max}^*$,
- Hallucination rate,
- Fraction of creative responses retained.

We test whether reductions in $\lambda_{\max}^*$ are associated with improved grounding, and whether there exists a “sweet spot” regime (analogous to the RFO ringing wedge) where creativity is preserved while hallucination remains controlled.

---

## 6.6 Limitations and Future Directions

We close by acknowledging key limitations:

- **Model dependence:**  
  Results may depend on the specific architecture and training of the models studied.

- **Proxy quality:**  
  The stability score $\lambda_{\max}^*$ is an approximation; different choices of graph construction or Jacobian estimation may change its behaviour.

- **Label noise:**  
  Hallucination labels may be noisy or subjective, especially in open-ended tasks.

These limitations suggest future work:

- Expanding to a broader range of models and tasks,
- Refining geometric estimators and connecting them more tightly to theory,
- Exploring online control schemes where $\lambda_{\max}^*$ is monitored and used to adapt decoding or grounding in real time.

---
