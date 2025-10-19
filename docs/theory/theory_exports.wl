(* Wolfram “run-once” script. Paste the notebook content provided by Wolfram verbatim: *)
(* --- GLOBAL PARAMETERS --- *)
SetDirectory@NotebookDirectory[];
α = 0.015; ω0 = 1; κ = 0.4;
βc = α; Φvals = Range[-3, 3, 0.05];
εvals = Table[10^-i, {i, 3, 9}];

(* ---------- 1. Eigenvalue Trajectories ---------- *)
βvals = Join[Range[0, 0.012, 0.001], Range[0, 0.006, 0.0002] /. 0 -> 0.012, Range[0.018, 0.03, 0.001]];
f[s_, β_] := -α s + β {{0, 1}, {1, 0}}.Tanh[s];
fdJac[s_, β_, ε_] := Table[(f[s + ε e, β] - f[s - ε e, β])/(2 ε), {e, IdentityMatrix[2]}];

dataEig = Flatten[
  Table[
    Module[{Jexact, Jeps, λa, λn},
      Jexact = -α IdentityMatrix[2] + β {{0, 1}, {1, 0}};
      λa = Eigenvalues[Jexact];
      Jeps = fdJac[{0, 0}, β, ε];
      λn = Eigenvalues[Jeps];
      <|"beta" -> β, "eps" -> ε,
        "re_eig1_analytic" -> Re[λa[[1]]],
        "re_eig2_analytic" -> Re[λa[[2]]],
        "re_eig1_fd" -> Re[λn[[1]]],
        "re_eig2_fd" -> Re[λn[[2]]]|>
    ],
    {β, βvals}, {ε, {1*10^-3, 1*10^-4, 1*10^-5, 1*10^-6, 1*10^-7}}
  ], 1];

If[!DirectoryQ["docs/data/theory"], CreateDirectory["docs/data/theory", CreateIntermediateDirectories -> True]];
If[!DirectoryQ["docs/assets/figures"], CreateDirectory["docs/assets/figures", CreateIntermediateDirectories -> True]];

Export["docs/data/theory/jacobian_eig_trajectories.json", dataEig, "JSON"];

pltEig = ListLinePlot[
  Table[{-α + β, -α - β} /. β -> βvals, {β, βvals}],
  PlotRange -> All,
  AxesLabel -> {"β", "Re[λ]"},
  PlotLegends -> {"λ₁", "λ₂"},
  Epilog -> {Dashed, Line[{{βc, -0.03}, {βc, 0.03}}]},
  PlotLabel -> "Eigenvalue trajectories near βc = 0.015",
  ImageSize -> 600];
Export["docs/assets/figures/jacobian_eig_trajectories.png", pltEig];

(* ---------- 2. FD Error vs ε ---------- *)
pts = {{{0, 0}, "linear"}, {{5, 5}, "saturated"}};
errorData = Flatten[
  Table[
    Module[{s = p[[1]], reg = p[[2]], Jexact, Jeps, Δ, λa, λn},
      Jexact = -α IdentityMatrix[2] + βc {{0, 1}, {1, 0}}.DiagonalMatrix[Sech[s]^2];
      Jeps = fdJac[s, βc, ε];
      Δ = Jeps - Jexact;
      λa = Eigenvalues[Jexact]; λn = Eigenvalues[Jeps];
      <|"epsilon" -> ε, "regime" -> reg,
        "dJ_fro" -> Norm[Δ, "Frobenius"],
        "max_abs_eig_err" -> Max[Abs[λa - λn]]|>
    ],
    {ε, εvals}, {p, pts}
  ], 2];

Export["docs/data/theory/jacobian_error_curve.json", errorData, "JSON"];

pltErr = ListLogLogPlot[
  {
    Select[errorData, #["regime"] == "linear" &][[All, {"epsilon", "dJ_fro"}]],
    Select[errorData, #["regime"] == "saturated" &][[All, {"epsilon", "dJ_fro"}]]
  },
  PlotLegends -> {"Linear", "Saturated"},
  AxesLabel -> {"ε", "‖ΔJ‖"},
  PlotLabel -> "FD Error vs ε",
  ImageSize -> 600];
Export["docs/assets/figures/jacobian_error_loglog.png", pltErr];

(* ---------- 3. Phase Surface Composite ---------- *)
regimes = {
  {"Stable", 1., 1.1111, 0.9},
  {"Balanced", 1., 1.0, 1.0},
  {"Divergent", 1., 0.9091, 1.1}
};

phaseData = Flatten[
  Table[
    Table[
      Module[{φ = Φ, Rφ, λφ, Vφ, αv = αr, βv = βr, regime = reg, ratio = r},
        Rφ = κ φ^2; λφ = Sqrt[(αv/βv) Rφ];
        Vφ = 0.5 ω0^2 φ^2 - αv Rφ + βv λφ^2;
        <|"phi" -> φ, "R_phi" -> Rφ, "lambda_phi" -> λφ, "V_phi" -> Vφ,
          "alpha" -> αv, "beta" -> βv, "alpha_over_beta" -> ratio,
          "regime" -> regime, "stability_flag" -> regime|>
      ],
      {Φ, Φvals}],
    { {reg, αr, βr, r}, regimes}
  ], 2];

Export["docs/data/theory/phase_surface_all.json", phaseData, "JSON"];

pltPhase = ListPointPlot3D[
  phaseData[[All, {"phi", "R_phi", "lambda_phi"}]],
  ColorFunction -> Function[{x, y, z}, ColorData["ThermometerColors"][z/2.1]],
  AxesLabel -> {"Φ", "R(Φ)", "λ(Φ)"},
  PlotLabel -> "Phase-Surface Composite",
  ImageSize -> 600];
Export["docs/assets/figures/phase_surface_composite.png", pltPhase];

(* ---------- 4. README summary ---------- *)
Export["docs/data/theory/README.md",
 "# Resonance Geometry — Theory Data Exports\n\n" <>
 "| File | Description | Figure |\n|------|--------------|---------|\n" <>
 "| jacobian_eig_trajectories.json | Eigenvalue trajectories vs β | Fig. 3.5 |\n" <>
 "| jacobian_error_curve.json | Finite-difference error vs ε | Fig. 3.6 |\n" <>
 "| phase_surface_*.json | Phase-space potential surfaces (3 regimes + combined) | Fig. 3.7 |\n",
 "Text"];
