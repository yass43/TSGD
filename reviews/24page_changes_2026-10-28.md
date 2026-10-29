# Shortening from 29 to 24 pages

The final count includes references and appendices. Font size and margins are unchanged.

## Cuts and consolidation

- Removed the introductory estimator diagram; Section 2 still derives TSGD and states all three estimator updates.
- Removed the sinusoidal simulation and its figure from the compiled paper. The fixed-variance/variable-similarity construction, lemma, proof, and exact recursions remain.
- Consolidated Appendix B's four cases into exact transport and stochastic transport. Retained the tilted-objective interpretation, GD convergence argument, initialization sharpness, mean-square convergence proof, conservation law, and interaction formula.
- Moved the least-squares residual calculation from Appendix B to the common experiment description in Appendix D.
- Placed the two appendix least-squares plots side by side. All four least-squares experiments remain; no simulation data or plot assets were modified.
- Removed the extended optimizer-error appendix from the compilation. Section 5 still contains the method comparison table, connections, practical qualifications, and motivations for combining estimators.
- Removed the arithmetic comparisons between optimized upper bounds. Retained initialization cost, equal-budget iteration counts, oracle complexity bounds, and a concise common-class lower-bound corollary with proof.
- Updated affected references and removed claims tied to the deleted numerical experiment.

## Retained safeguards

The sampled/controlled initialization distinction, noise-magnitude dependence through initialization, finite-horizon qualification, restriction of the lower bound to zero-respecting algorithms, and distinction between upper bounds and actual algorithm performance remain explicit.

## Verification and recovery

- Successful latexmk build: 24 pages; no undefined references or LaTeX layout warnings.
- Inspected the consolidated quadratic analysis, paired plots, and final complexity page.
- The preceding 29-page PDF and source files are preserved in `before_24page_edit_2026-10-28/`.
- Omitted experiment scripts, data, figure files, and the optimizer appendix source remain in the repository. No bibliography entries were edited.
