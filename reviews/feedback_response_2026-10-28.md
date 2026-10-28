# Response to the annotated main.pdf

All 23 written comments were addressed. Page numbers below refer to the annotated original, preserved as `main_feedback_2026-10-28.pdf` in this directory. Associated highlights were treated as part of their comments, not as separate requests.

| # | Original page | Problem | Resolution |
|---|---|---|---|
| 1 | 1 | The abstract foregrounded false convergence as a shortcoming. | Replaced that clause with “quantify how initialization and transport errors interact.” False convergence remains in the technical discussion. |
| 2 | 1 | Applicability beyond finite sums needed emphasis. | Wrote: “We stress that centered similarity applies to population objectives as well as finite sums.” |
| 3 | 1 | The explanation of the two error sources was awkward. | Explicitly distinguished error in the initial estimate from error in gradient changes, then stated that the former persists and the latter accumulates. Removed a redundant sentence. |
| 4 | 2 | The introduction's feature-variability qualification was unnecessary. | Removed it there; retained the Section 2.2 pointer and the substantive explanation in that section. |
| 5 | 2 | The third contribution sounded weak. | Foregrounded the construction of oracles with identical pointwise gradient distributions but different transport accuracy, and exact examples distinguishing the two algorithms. Retained experiments, comparisons, and connections without claiming a new method or rate. |
| 6 | 2 | “Collected” was unnatural. | Changed to “Detailed proofs and additional calculations appear in the appendices.” |
| 7 | 3 | The opening question lacked motivation. | Started with the need for a gradient estimate to determine the next update after reaching the new iterate, followed by the gradient-change identity. |
| 8 | 3 | The SGD description used an awkward comma. | Changed to “directly by drawing a fresh stochastic gradient.” |
| 9 | 3 | A1 did not display the defining Lipschitz-gradient inequality. | Added the inequality explicitly for every pair of points, without changing assumption numbering. |
| 10 | 3 | The explanation after the updates began with MSGD rather than SGD. | Added SGD's fresh-estimate description first, then explained MSGD and TSGD in equation order. |
| 11 | 3 | Bounded variance appeared without its motivation for SGD. | Opened Section 2.2 with “SGD is typically analyzed under a bounded-variance assumption.” |
| 12 | 4 | Finite-sum normalization used n^-1 rather than 1/n. | Used the fraction consistently in Section 2.2 and the corresponding appendix passages. |
| 13 | 4 | The finite-sum and least-squares examples made Section 2.2 too heavy. | Kept a short finite-sum interpretation and moved its displayed gradient-difference calculation to Appendix C.3. Retained least squares, which supports the experiments, and tightened its interpretation. |
| 14 | 4 | Section 2.3 needed a connection to the previously announced error sources. | Added an opening recalling initialization and transport error before making the decomposition precise. |
| 15 | 5 | “Keeping track” did not explain the purpose of the examples. | Reframed the section around when accurate changes favor TSGD and fresh estimates favor SGD, using the true-gradient stationarity measure. |
| 16 | 7 | “Need not” weakened the false-convergence explanation. | Stated directly that the estimator tends to zero while the expected squared true-gradient norm stays positive. |
| 17 | 7 | “Obstruction” was unsuitable wording. | Explained directly that adding a linear term leaves gradient differences unchanged; aligned the appendix wording. |
| 18 | 10 | SVRG's reference point should be called an anchor. | Used “anchor point” for SVRG, retaining the explanation that it is fixed within an epoch. |
| 19 | 11 | Momentum lag needed a clearer description. | Specified that its estimate can lag behind the current gradient, especially when the gradient changes rapidly. |
| 20 | 12 | The mixing-parameter paragraph sounded generic and unnatural. | Explained the concrete choice: accurate differences preserve information, fresh gradients can correct accumulated error, and refresh decisions trade this benefit against fresh-estimation cost and noise. |
| 21 | 12 | The exact-transport exception distracted from the complexity discussion. | Removed that sentence while retaining the caveat that the comparison does not establish an epsilon^-4 lower bound for pure TSGD. |
| 22 | 12 | The conclusion's magnitude-versus-variation sentence was awkward. | Stated when transport is effective: sufficiently accurate initialization and small noise variation, even with very noisy individual gradients. |
| 23 | 12 | The conclusion did not explicitly argue for studying the mechanism alone. | Added that the known recursion deserves study on its own, making its errors explicit and clarifying when to combine it with other mechanisms. |

## Additional correction

Section 3.4 previously said Appendix B.6 checks “the four regimes” numerically. Its experiment actually varies two nonzero initialization-error levels and two nonzero similarity levels, rather than reproducing all four exact/inexact cases. The pointer now accurately says it varies initialization accuracy and similarity on least squares and compares observed residuals with exact limits.

## Verification

- Compiled successfully with latexmk: 37 pages; main text ends on page 12.
- No LaTeX warnings, undefined references, or overfull/underfull box warnings in the final build log.
- Visually checked revised pages 1, 3, 4, and 12.
- Preserved the original annotated PDF; no bibliography changes were made.
