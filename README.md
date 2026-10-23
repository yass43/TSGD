# TSGD: The Sister of SGD

The manuscript presents transported SGD as a fundamental gradient-estimation
mechanism. The main text contains six sections; Appendices A--E preserve the
detailed proofs, examples, and algorithmic identities.

## Build

From this directory, run:

```sh
latexmk -pdf main.tex
```

Or run `make`. The project uses standard TeX Live/MiKTeX packages and requires
no custom venue class. Python is not required to compile: the numerical figure
is included as a PDF. If an editor also builds on save, use a separate output
directory for a manual build to avoid simultaneous writes to auxiliary files:

```sh
latexmk -pdf -outdir=build main.tex
```

## Layout

- `main.tex`: document assembly, title, authors, and abstract.
- `preamble.tex`: packages and document-wide formatting.
- `macros.tex`: project-specific notation.
- `theorems.tex`: theorem-like environments.
- `sections/01_introduction.tex`: motivation and contributions.
- `sections/02_estimators.tex`: SGD, MSGD, TSGD, similarity, and error identity.
- `sections/03_regimes.tex`: examples, numerical figure, and four error regimes.
- `sections/04_theory.tex`: main theorem and comparison with SGD.
- `sections/05_modern_optimizers.tex`: estimator relationships and combinations.
- `sections/06_conclusion.tex`: concluding perspective.
- `appendix/`: full convergence proofs (A), quadratic dynamics (B), noise
  assumptions (C), detailed examples and numerical protocol (D), and oracle
  costs and algorithmic details (E, assembled from two files).
- `references.bib`: bibliography.
- `figures/fundamental_estimators.tex`: conceptual vector diagram using LaTeX.
- `figures/similarity_experiment.pdf`: numerical figure included in the paper.
- `scripts/`: experiment source, dependencies, and reproduction instructions.
- `REWRITE_MAP.md`: locations of the results from the previous manuscript.
- `archive/pre-rewrite-2026-10-23.zip`: complete source snapshot before the rewrite.

## Reproduce the numerical figure

From this directory, with the dependencies in `scripts/requirements.txt` installed:

```sh
python scripts/similarity_experiment.py
```

The default run uses 100,000 independent trajectories per curve and seed
20261023. It writes the PDF/PNG, per-iteration CSV, and parameter/diagnostic JSON
in `figures/`. Appendix D documents the shared, charged antithetic initialization,
evaluation costs, fixed stepsize, and analytical checks. The experiment compares
actual mean squared true gradients at equal oracle budgets; it is not a claim
that one method or frequency ordering is best for every parameter choice.
