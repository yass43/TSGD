# Similarity experiment

Run from the paper directory:

```text
python -m pip install -r scripts/requirements.txt
python scripts/similarity_experiment.py
```

The default experiment uses 100,000 independent trajectories per curve,
seed 20261023, and a budget of 200 stochastic-gradient evaluations per method.
It generates `figures/similarity_experiment.pdf` and `.png`, the per-iterate
mean/standard-error data in `.csv`, and parameters, environment versions,
analytical checks, and final tail summaries in `.json`.

The oracle is `h(x, phase) = x + sqrt(2) * cos(omega*x + phase)`.
All methods start at `x0=1` and use the same constant stepsize `eta=0.1`.
Each gets an exact initial estimate by averaging the oracle at phases
`phase0` and `phase0+pi`, charged as two pointwise evaluations. This requires
the explicit antithetic sampling available in this controlled example; it
does not assume a generic independent batch of size two is exact.
SGD uses this common estimate for its first update and fresh single-sample
estimates thereafter. TSGD spends two evaluations on each subsequent update,
using the same fresh phase at the old and new iterates. No unused final
transport is computed. The initial point is plotted at zero evaluations.

The shaded regions are Monte Carlo means plus/minus two estimated standard
errors, not population prediction intervals or simultaneous confidence bands.
The representative SGD simulation uses `omega=1`; its exact distribution and
second-moment recursion are frequency-independent. The script checks this
curve against the exact recurrence and checks `omega=0` TSGD against GD.
It retains all trajectories, including rare large errors, and reports the
largest 0.1 percent's share of the final squared-gradient sum.

Optional flags: `--runs`, `--budget`, and `--seed`. The manuscript reports the
default experiment; changing these flags overwrites the same artifact paths.
