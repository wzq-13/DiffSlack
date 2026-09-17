# DiffSlack

**DiffSlack: Learning under Nonlinear Inequality Constraints via Learnable Slack Variables**

DiffSlack is a learning-based framework for nonlinear inequality-constrained prediction. This repository applies it to autonomous-vehicle trajectory planning in cluttered environments and includes data generation, NMPC supervision, two-stage training, evaluation, diagnostic tools, and several baselines.

The current implementation predicts a 40-waypoint trajectory. DiffSlack augments each waypoint with five learnable slack variables and applies an adaptive-depth weighted Gauss--Newton projection. The projection keeps the original stop-gradient training behavior. ENFORCE is provided as a Fischer--Burmeister-based comparison.

## Highlights

- Learnable slack initialization and projection for nonlinear inequality constraints.
- Four DiffSlack initialization modes: `learned`, `zero`, `constant`, and `analytic`.
- Two ENFORCE multiplier initialization modes: `zero` and `learned`.
- Weighted projection metric with separately configurable trajectory and slack weights.
- Two-stage training: soft-constraint pretraining followed by projection-aware training.
- End-to-end and kernel-level timing measurements.
- Exact-collision failure diagnosis with larger iteration budgets and optional single-sample overfitting.
- Random map generation with potential fields, shortest paths, and NMPC supervision.
- A self-contained nonlinear constrained-regression demo comparing MLP, soft penalty, ENFORCE, and DiffSlack.

## Repository Structure

```text
DiffSlack/
├── DiffSlack.py                  # DiffSlack training and evaluation entry point
├── ENFORCE.py                   # ENFORCE training and evaluation entry point
├── IL.py                        # Imitation-learning soft/pure baseline entry
├── IL_hard.py                   # Hard-constraint IL baseline entry
├── data_generator.py            # Map, potential-field, and NMPC-label utilities
├── demo.py                      # Visual introduction to the regression problem
├── demo_train.py                # Standalone four-method regression experiment
├── environment.yml              # Reproducible Conda environment
├── globalvar.py                  # Vehicle and planning-space parameters
├── IL.json                      # NMPC-label file list used by IL loaders
├── configs/
│   ├── DiffSlack.yaml
│   ├── ENFORCE.yaml
│   ├── DC3.yaml
│   ├── IL.yaml
│   └── IL hard.yaml
├── models/                      # Projection layers, networks, and trainers
├── DataLoader/                  # Raw-map and NMPC-label dataset loaders
├── others/                      # NMPC, Hybrid A*, and supporting planners
└── utils/                       # Constraints, geometry, metrics, and visualization
```

The repository contains DC3 trainer/configuration code, but it does not currently include a standalone top-level DC3 entry script.

## Installation

The exported environment uses Python 3.9, PyTorch 2.6.0 with CUDA 11.8, CasADi, Shapely, SciPy, MMEngine, and the remaining research dependencies.

```bash
conda env create --name diffslack --file environment.yml
conda activate diffslack
```

The CUDA build in `environment.yml` should be adjusted if the host uses a different CUDA/driver stack. CPU execution is sufficient for the small regression demo, while the full trajectory experiments are substantially faster on a GPU.

## Quick Start: Standalone Regression Demo

The demo does not require the trajectory dataset.

To visualize the nonlinear feasible interval and noisy labels:

```bash
python demo.py
```

To train and compare an unconstrained MLP, a soft-penalty model, ENFORCE, and DiffSlack:

```bash
python demo_train.py --device auto --output-dir demo_results
```

A short smoke run can be launched with:

```bash
python demo_train.py \
  --device cpu \
  --n-train 256 \
  --n-test 256 \
  --stage1-epochs 5 \
  --stage2-epochs 5 \
  --output-dir demo_results_smoke
```

The demo writes its configuration, per-seed metrics, summary tables, model checkpoints, Jacobian timing results, and `predictions.png`/`predictions.pdf` to the selected output directory. Run `python demo_train.py --help` for all options, including analytical versus autodiff Jacobians.

## Dataset

### Download

The datasets are available from this [Google Drive folder](https://drive.google.com/drive/folders/1g11mi35CIDDpZGkraySVMox_5b98mBt0?usp=drive_link). It contains three archives:

- `V2_0-100k.tar.zst`: raw trajectory-planning samples with indices from 0 to 99,999;
- `V2_100k-200K.tar.zst`: raw trajectory-planning samples with indices from 100,000 to 199,999;
- `data-NMPC.tar.zst`: the NMPC expert dataset used by the imitation-learning baselines.

Extract both `V2` archives into the same raw-data directory, for example:

```text
dataset/
├── 0.npz
├── 1.npz
└── ...
```

Each raw `.npz` sample contains at least:

- `distance_map`: the potential/distance field;
- `obstacles_vertices`: eight quadrilateral obstacles with shape `(8, 4, 2)`;
- `target`: the goal position with shape `(2,)`.

### Generate Planning Data

`data_generator.py` generates random environments, a navigation graph, the potential field, and shortest-path diagnostics:

```bash
python data_generator.py
```

The current `__main__` block generates 2,000 samples beginning at index 0 and writes them to `./dataset/`. Change `num` and `begin` near the bottom of the file for a different range. A multiprocessing implementation is included but commented out.

The primary programmatic APIs are:

```python
from data_generator import generate_map_data, generate_nmpc_supervision

sample = generate_map_data()

label = generate_nmpc_supervision(
    "dataset/0.npz",
    initial_path="potential_field",
    save_path="dataset_NMPC-label/0.npz",
)
```

Passing `initial_path=None` preserves the NMPC planner's straight-line initialization. Passing `"potential_field"` uses the stored shortest path, or reconstructs it from the distance map for older files.

### Generate NMPC Labels

The original batch-labeling helper is `generate_labels_for_IL(index)` in `others/test_NMPC.py`. Its input and output directories are currently hard-coded:

```text
/home/qian/dataset_V7/
/home/qian/dataset_V7_NMPC-label/
```

Update these paths before using that helper, or use `data_generator.generate_nmpc_supervision` as shown above. `IL.json` must list the filenames of the successfully generated NMPC labels used by `My_Dataset_IL`.

## Full Trajectory Experiments

### Important Path and Split Settings

The DiffSlack and ENFORCE entry points currently construct `My_Dataset(..., length=200000)`. The directory must therefore contain files named `0.npz` through `199999.npz`, unless the `length` argument and split logic are adjusted in the entry script.

DiffSlack currently defaults to `/home/qian/dataset_V7/`; always pass `--data-dir` on another machine. ENFORCE defaults to `./dataset/`.

Both entry points default to loading a stage-1 checkpoint. The supplied YAML files use `begin_epoch: 300`, so their default workflow resumes stage-2 projection training. For a fresh two-stage run, set `begin_epoch: 0` in the corresponding YAML and set the entry script's `--load-dir` default to `None`.

### DiffSlack

Resume stage-2 training:

```bash
python DiffSlack.py \
  --data-dir ./dataset \
  --load-dir save_dir/DiffSlack/epoch_299.pth
```

Evaluate a checkpoint:

```bash
python DiffSlack.py \
  --data-dir ./dataset \
  --load-dir save_dir/DiffSlack/epoch_299.pth \
  --test-only
```

Override slack initialization from the command line:

```bash
# learned | zero | analytic
python DiffSlack.py --data-dir ./dataset --initialization analytic --test-only

# A constant initialization additionally accepts --constant.
python DiffSlack.py --data-dir ./dataset \
  --initialization constant --constant 0.1 --test-only
```

The projection metric is configured in `configs/DiffSlack.yaml`:

```yaml
w_traj: 5.0
w_slack: 1.0
damping: 1.0e-4
max_depth: 50
inference_tol: 1.0e-3
```

The current implementation intentionally exposes no backward-mode option: training uses the original stop-gradient projection behavior.

### ENFORCE

Resume stage-2 training or evaluate a checkpoint:

```bash
python ENFORCE.py \
  --data-dir ./dataset \
  --load-dir save_dir/enforce/epoch_299.pth

python ENFORCE.py \
  --data-dir ./dataset \
  --load-dir save_dir/enforce/epoch_299.pth \
  --test-only
```

Select the Fischer--Burmeister multiplier initialization with:

```bash
python ENFORCE.py --data-dir ./dataset --initialization zero --test-only
python ENFORCE.py --data-dir ./dataset --initialization learned --test-only
```

`models/ENFORCE_Trainer.py` currently sets `DEVICE = torch.device("cpu")`. Change that assignment if GPU execution is desired.

### IL Baselines

`IL.py` and `IL_hard.py` retain experiment-specific dataset and checkpoint paths in their source. Update those paths before running:

```bash
python IL.py
python IL_hard.py
```

`models/DC3_Trainer.py` and `configs/DC3.yaml` provide the DC3 implementation, but a top-level DC3 launcher is not included in this release tree.

## Timing and Failure Diagnosis

Both primary entry points can report a detailed projection timing breakdown:

```bash
python DiffSlack.py --data-dir ./dataset --test-only --profile-timing
python ENFORCE.py --data-dir ./dataset --test-only --profile-timing
```

This adds Jacobian construction, Cholesky decomposition, and linear-solve/update timing without including the extra instrumented pass in end-to-end latency.

DiffSlack also supports exact-collision failure diagnosis:

```bash
python DiffSlack.py \
  --data-dir ./dataset \
  --test-only \
  --failure-diagnosis \
  --diag-i-max 200
```

The diagnostic pipeline distinguishes surrogate mismatch, iteration-limited failures, prediction-route-limited failures, and unresolved failures. It writes `failure_sample_ids.pt`, `failure_diagnosis.pt`, and diagnostic figures under the run's log directory. Add `--disable-sample-overfit` to skip the single-sample overfitting stage.

To rerun diagnosis only on a previously saved collision subset:

```bash
python DiffSlack.py \
  --data-dir ./dataset \
  --failure-sample-ids logs/<run>/failure_sample_ids.pt \
  --diag-i-max 200
```

## Outputs

DiffSlack and ENFORCE create run directories using the initialization mode and seed:

```text
logs/<run-name>/<run-name>_<seed>/
save_dir/<run-name>/<run-name>_<seed>/
```

Use `--run-name` to override the generated name. A run may contain:

- the resolved YAML configuration;
- a copy of `utils/prob.py`;
- TensorBoard event files;
- `test_results_soft.txt` and `test_results_hard.txt`;
- visualization images;
- model/optimizer/scheduler checkpoints.

Typical reported metrics include collision rate, path length, smoothness, curvature feasibility, final distance to the target, waypoint-distance violations, projection depth, projection displacement, and latency.

The trainer classes also expose `test_visualization(...)` and `save_path_data(...)` for custom evaluation scripts. Exported trajectories are written as `batch_<index>.npy` files in the requested directory.

## Reproducibility Notes

- Random seeds are set in the entry scripts and YAML configurations.
- Planning-space dimensions, vehicle geometry, safety margin, turning radius, and obstacle count are defined in `globalvar.py`.
- DiffSlack predicts 280 values: 40 waypoints with `[x, y, s1, s2, s3, s_curvature, s_distance]` at each waypoint.
- ENFORCE and the path-only baselines predict 80 values: 40 `(x, y)` waypoints.
- `torch.compile` is used by the training projection path, so use a compatible PyTorch version.
- The evaluation code performs warm-up iterations before collecting timing measurements.

## Acknowledgements

This project builds on ideas and utilities from:

- [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics)
- [ENFORCE](https://github.com/process-intelligence-research/ENFORCE)
- [DC3](https://github.com/locuslab/DC3)
