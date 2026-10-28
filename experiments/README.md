# Pure TSGD: four least-squares mechanism experiments

These are **iteration-based illustrations, not computational-efficiency comparisons**.
All methods use the same constant stepsize (default `eta=0.5`), with no tuning,
refreshes, momentum, clipping, or parameter selection based on outcomes.
The manuscript and its existing figures are not modified by this script.

## Cluster: existing Conda environment and Slurm

The supplied `FL_EF21_Second_Order-main (2).zip` uses **`FL_EF21_ENV`**,
Slurm `sbatch`, and Conda activation through `conda info --base`. Older
scripts identify the installation as `/ibex/user/maziany/miniforge`.
We reuse that environment, with the same path as a fallback if `conda` is
not on PATH. The archive does not contain a complete Conda environment
export, so exact installed Python/package versions cannot be inferred.

Copy this `experiments` directory to the cluster (or transfer the paper
repository), then run from that directory:

```bash
bash submit_least_squares.sh smoke --dry-run  # prints command; submits nothing
bash submit_least_squares.sh smoke           # 200 trajectories, 50 iterations
bash submit_least_squares.sh full            # 4,000 trajectories, 400 iterations
# Optional larger Monte Carlo run, same model and parameters:
bash submit_least_squares.sh full --runs 20000
```

The wrapper works from any current directory. It creates the log directory
before submitting, uses `sbatch --parsable`, and sets the job working
directory to this experiment folder. Each submission has separate results
under `results/cluster/<profile>_<jobid>/` and logs under
`results/cluster/logs/`. Monitor with `squeue -j <jobid>` and read the
corresponding `.out` / `.err` files. No existing results are replaced by
the default job output paths. Explicitly passing `--output` overrides this.

The job runs the tests first, then the experiment and plotting on the CPU.
It requests **one CPU, 4 GB RAM, 5 minutes for smoke / 30 minutes for full**;
no GPU, dataset download, array, or dependent finalize job is needed for this
small experiment. These time limits are starting allocations, not measurements
on the cluster. No jobs are submitted by the Python script itself.

Optional overrides, matching the existing environment convention:

```bash
CONDA_ENV_NAME=FL_EF21_ENV bash submit_least_squares.sh full
CONDA_BASE=/ibex/user/maziany/miniforge bash submit_least_squares.sh smoke
TSGD_TIME=01:00:00 TSGD_MEM=8G bash submit_least_squares.sh full --runs 50000
```

`TSGD_CPUS` overrides allocated CPUs; BLAS/OpenMP thread limits follow the
allocation. Increasing CPUs will not parallelize the independent scenario
loop. `PYTHON_BIN` optionally selects a Python executable after activation.
There is no hard-coded remote project path or requested account/partition,
consistent with the supplied submission scripts. Use LF line endings for
`.sh` and `.slurm` files; `.gitattributes` enforces this on Git checkouts.

**No automatic installation:** only NumPy >=1.24 and Matplotlib >=3.7 are
needed, already included in the supplied project's requirements. Neither
submission nor execution modifies the environment. If an import is missing,
activate the environment and explicitly install this folder's requirements.
Actual Python/package versions, Conda environment, Slurm job ID, and thread
limits are recorded with each run. The local checks used NumPy 2.2.6 and
Matplotlib 3.10.5; other allowed versions have not all been tested here.

## Local run

From the repository's parent directory:

```powershell
python -m pip install -r TSGD/experiments/requirements.txt
python TSGD/experiments/least_squares.py
python -m unittest discover -s TSGD/experiments -v
```

Or from inside `TSGD/experiments`, run `python least_squares.py`.
The local code does not require Conda or Slurm; dependency minimums match
the NumPy/Matplotlib requirements of the supplied codebase.
Default: 4,000 independent trajectories per scenario, 400 iterations, seed 20261028.
For a quick check: `python least_squares.py --runs 200 --steps 100 --output results/smoke`.
Increase `--runs` to reduce Monte Carlo uncertainty; do not select seeds for favorable results.

Outputs in `results/` (or `--output PATH`):

- `least_squares_A.pdf` / `.png` through `least_squares_D.pdf` / `.png`: four independent plots, each with one legend and no explanatory footer or inset.
- `curves.csv`: every iteration's true-gradient, estimator, and tracking-error
  squared norms, their Monte Carlo standard errors, and exact expectations.
- `settings_and_summary.json`: parameters, software versions, initialization
  assumptions, exact limiting errors, and cancellation-check discrepancy.

## Replot an existing job (no resimulation)

From `experiments` in your activated environment:

```bash
python plot_least_squares.py --run-dir results/cluster/full_54712913
# Optional display controls:
python plot_least_squares.py --run-dir results/cluster/full_54712913 --max-iter 60 --exact
python plot_least_squares.py --run-dir results/cluster/full_54712913 --full-horizon
```

The command reads only the saved CSV/JSON and writes four separate PDF/PNG figures plus
`caption.txt` in the run's `replot/` subfolder. Original figures and raw results
are preserved by default. `--output PATH` selects another figure directory.
No Slurm job or new Monte Carlo run is needed. Default display: iterations 0–80;
the vertical limits are fitted to the visible data, not the entire trajectory.
`--no-bands` hides uncertainty shading. Existing combined figures from earlier
versions are left untouched; open the new `least_squares_A` through `least_squares_D` files.

Plots contain only curves, axes, and one legend each, with no titles. A identifies
initial batches; B distinguishes SGD markers from the dashed TSGD estimator;
C groups the coincident transport curves; D lists the four (delta, E0) combinations.
The sampled/controlled distinction and interpretation of C's transport-noise
residual remain in the notebook and separate `caption.txt`, not inside the images.

New simulation jobs automatically use this renderer. `--plot-max-iter` changes
only the figure horizon, never the number of simulated or saved iterations.

## Inspect a completed run in a notebook

Open `inspect_results.ipynb` in Jupyter or VS Code, select the existing
`FL_EF21_ENV` kernel, and run the cells in order. The first cell selects
`results/cluster/full_54712913` by default. Edit `RUN_DIR` there for another
job or a downloaded local results directory. It lists available runs and
fails explicitly if the selected run is missing; it does not silently show
another experiment.

The notebook rebuilds the updated figure, each panel separately, sampled
initialization errors, the false-convergence diagnostics, cancellation checks,
and exact finite-time/asymptotic comparisons. Set `SHOW_EXACT`, `SHOW_BANDS`,
and `MAX_ITER` to inspect details; `OVERVIEW_MAX_ITER` controls the overview.
It imports the shared `plot_least_squares.py`, so copy that file along with the
updated notebook. It reads the CSV/JSON and never reruns the
experiment, submits jobs, or overwrites results. No pandas or widget packages
are required beyond an existing notebook frontend/kernel and NumPy/Matplotlib.

On the cluster, the confirmed environment is
`/ibex/user/maziany/conda-environments/FL_EF21_ENV` (job 54712913 used
NumPy 2.4.3 and Matplotlib 3.10.9). If your notebook frontend does not list it,
register its kernel from an activated environment, assuming `ipykernel` is installed:

```bash
conda activate /ibex/user/maziany/conda-environments/FL_EF21_ENV
python -m ipykernel install --user --name FL_EF21_ENV --display-name "Python (FL_EF21_ENV)"
```

No kernel installation is performed automatically. To inspect locally,
copy the completed result folder and select a local Python notebook kernel;
the analysis does not require access to the remote environment.

## Model

We use **vector-response** least squares, not ordinary one-label regression:

\[
f(x)=\tfrac12\mathbb E\|Ax-b\|^2,\quad
A=\sqrt v\,H^{1/2},\quad b=Ax^\star+s\varepsilon,
\quad \varepsilon\sim\mathcal N(0,I).
\]

Here `x*=0`, `v=1-kappa` or `1+kappa` with equal probability, and the feature
and label noises are independent. `H` is diagonal with 10 eigenvalues evenly
spaced between 0.1 and 1. `x0` is proportional to the all-ones vector and
normalized so `||H x0||^2=1`.

\[
f(x)=\tfrac12x^\top Hx+\tfrac d2s^2,\qquad
\nabla f(x)=Hx,\qquad L=1,
\]
\[
h(x)=vHx-s\sqrt v\,H^{1/2}\varepsilon,\qquad
h(x)-h(y)=vH(x-y),
\]
\[
\mathbb E\|h(x)-Hx\|^2=\kappa^2\|Hx\|^2+s^2\operatorname{tr}(H),
\quad
\mathbb E\|[h(x)-h(y)]-H(x-y)\|^2=\kappa^2\|H(x-y)\|^2.
\]

The smallest global centered similarity constant is `delta=kappa` because
`||H||=1`. Varying `kappa` or `s` leaves the population gradient and curvature
unchanged. For `kappa>0`, pointwise gradient-noise variance is unbounded in `x`:
`s^2` is NOT a global bounded-variance constant.
This intentionally structured distribution isolates the mechanisms; it is
not a claim about typical real-world feature distributions.

## Panels and initialization

| Panel | Settings | Initialization and purpose |
|---|---|---|
| A | `s=1`, `delta=.05` | SGD uses one sample; TSGD uses actual iid batches `B0=1,128,512`. Shows noisy fresh gradients versus accurate changes, and sensitivity to initial quality. |
| B | `s=0`, `delta=.9` | SGD uses one sample; TSGD uses an actual batch `B0=128`. Shows SGD's vanishing true gradient and TSGD's vanishing estimator despite a nonzero true gradient. |
| C | `delta=.2`, `s=0,1,5` | **Controlled exact initialization for both methods.** Feature draws are matched across noise levels. TSGD trajectories coincide up to roundoff. |
| D | `s=1`, `delta=.05,.9`, `E0=.001,.1` | **Controlled**, independent Gaussian initial error with covariance `(E0/d)I`. Four TSGD curves isolate initial quality and similarity. |

For actual sampled initialization in A-B:

\[
g_0=B_0^{-1}\sum_{j=1}^{B_0}h(x_0;v_j,\varepsilon_j),\qquad
E_0=\frac{\kappa^2\|Hx_0\|^2+s^2\operatorname{tr}(H)}{B_0}.
\]

The initialization samples are independent of subsequent samples. C-D do not
simulate acquisition of their initial information and must not be presented
as equivalent-cost alternatives to A-B. Larger initial batches in A are not
free; their cost is deliberately not the question illustrated by this figure.
SGD and TSGD both take `x_{k+1}=x_k-eta*g_k`. SGD then obtains a fresh
gradient, whereas TSGD adds a difference evaluated at both endpoints with
the same newly drawn `(v, epsilon)`. Even with prescribed `g0` in C, all SGD
estimates after initialization are fresh stochastic gradients.

The panels share a common stable stepsize, but make no claims about optimal
tuning or universal superiority. In A, poorly initialized TSGD can be worse
than SGD despite very accurate transport. Small delta improves the theoretical
guarantee, not necessarily every finite-time realization. At a fixed nonzero
delta, even exactly initialized pure TSGD can have a residual.

## Mathematical checks

For eigenvalue `lambda_i`, let `X_i=E[x_i^2]`, `C_i=E[x_i g_i]`, and
`M_i=E[g_i^2]`. The exact TSGD moment recursions are

\[
X_i^+=X_i-2\eta C_i+\eta^2M_i,\quad
C_i^+=(1-\eta\lambda_i)(C_i-\eta M_i),\quad
M_i^+=\rho_i M_i,
\]
\[
\rho_i=(1-\eta\lambda_i)^2+\eta^2\kappa^2\lambda_i^2.
\]

For SGD, after updating `X` as above, fresh sampling gives

\[
C_i^+=\lambda_i X_i^+,\qquad
M_i^+=(1+\kappa^2)\lambda_i^2X_i^++s^2\lambda_i.
\]

These formulas cover sampled AND controlled initial estimates. The code
tracks exact finite-time expectations, rather than comparing experiments
to a loose convergence upper bound. Under the model's mean-square stability
condition `0 < eta < 2/((1+kappa^2)L)`, the TSGD residual is

\[
G_\infty=E_0+\sum_i
\frac{\eta\kappa^2\lambda_i}{2-\eta(1+\kappa^2)\lambda_i}
\mathbb E[g_{0,i}^2],\qquad M_k\to0.
\]

Independent trajectories are the Monte Carlo units; coordinates within a
trajectory share the feature draw `v` and are NOT treated as independent runs.
Shading is mean +/- two pointwise standard errors, not simultaneous confidence
bands. Nonpositive values are masked on log axes; positive values are not
floored, and raw CSV values are unchanged. Panel C uses matched random streams for cancellation; different
markers identify its overlapping TSGD curves.

Tests check shared-noise cancellation, exact recovery of GD, persistence of
initialization error, actual batch-initialization variance, Monte Carlo versus
exact moments, the quadratic conservation law, and false convergence.
