# DiffSlack

**DiffSlack: Learning under Nonlinear Inequality Constraints via Learnable Slack Variables**

DiffSlack is a learning-based trajectory planning project for constrained autonomous driving scenarios. The repository includes data generation, NMPC labeling, several learning baselines, DiffSlack training and evaluation, and CARLA-based trajectory tracking experiments.

## Overview

The project follows this pipeline:

1. Install dependencies from `requirements.txt`.
2. Generate planning scenes with `data_generator.py`.
3. Use NMPC to generate trajectory labels for imitation learning.
4. Train and evaluate DiffSlack and baseline methods.
5. Export planned trajectories with each method's `save_path_data` function.
6. Track the saved trajectories in CARLA using the provided map and tracking script.

## Repository Structure

```text
DiffSlack/
├── data_generator.py          # Random scene and obstacle data generation
├── others/test_NMPC.py        # NMPC planner and label generation
├── DiifSlack.py               # DiffSlack training / testing entry
├── ENFORCE.py                 # ENFORCE baseline entry
├── DC3.py                     # DC3 baseline entry
├── IL.py                      # IL soft / pure baseline entry
├── IL_hard.py                 # IL hard-constraint baseline entry
├── configs/                   # YAML configs for each method
├── models/                    # Network definitions and trainers
├── DataLoader/                # Dataset loaders
├── utils/                     # Geometry, visualization, and planning utilities
├── carla/
│   ├── Map.umap               # CARLA map asset
│   └── track.py               # CARLA trajectory tracking and evaluation
└── save_dir/                  # Example checkpoints
```

## 0. Install Dependencies

Install the required Python packages before generating data or running experiments:

```bash
pip install -r requirements.txt
```

Using a dedicated virtual environment or conda environment is recommended.

## 1. Generate Data

First generate planning environments:

```bash
python data_generator.py
```

By default, generated samples are saved under:

```text
./dataset/
```

Each `.npz` sample contains the planning scene information used by the learning methods, including obstacle geometry, target position, and map-related features. The number of generated samples and multiprocessing behavior can be adjusted near the bottom of `data_generator.py`.

## 2. Generate NMPC Labels

NMPC is used to solve trajectory optimization problems and produce labels for imitation learning.

The relevant script is:

```bash
python others/test_NMPC.py
```

The main labeling function is:

```python
generate_labels_for_IL(index)
```

It loads raw planning data, runs the NMPC planner, and saves successful labels in `.npz` format. In the current code, the paths are configured inside `others/test_NMPC.py`, for example:

```text
/home/qian/dataset_V7/
/home/qian/dataset_V7_NMPC-label/
```

If you want to run the full IL pipeline in this repository layout, update these paths to match your local dataset directory, or mirror the expected directory structure.

## 3. Train and Validate

Each method has its own entry script and config file.

| Method | Entry | Config |
| --- | --- | --- |
| DiffSlack | `DiifSlack.py` | `configs/APF_hard.yaml` |
| ENFORCE | `ENFORCE.py` | `configs/ENFORCE.yaml` |
| DC3 | `DC3.py` | `configs/DC3.yaml` |
| IL Soft / Pure | `IL.py` | `configs/IL.yaml` |
| IL Hard | `IL_hard.py` | `configs/IL hard.yaml` |

Example:

```bash
python DiifSlack.py
```

Inside each entry script, the trainer exposes the common workflow:

```python
trainer.train(...)
trainer.test(...)
trainer.test_visualization(...)
trainer.save_path_data(...)
```

For training, uncomment the corresponding `trainer.train(...)` line. For validation or testing, use `trainer.test(...)` or `trainer.test_visualization(...)`.

The scripts save logs, copied configs, and checkpoints under:

```text
logs/
save_dir/
```

The dataset split is currently defined directly inside each entry script. For example, DiffSlack, ENFORCE, and DC3 use `My_Dataset(data_dir='./dataset/', length=200000)`, while IL methods use NMPC-labeled data through `My_Dataset_IL`.

## 4. Export Trajectory Data

Before running CARLA tracking, each planning method needs to export its planned trajectories. This is done by calling the corresponding trainer's `save_path_data` function.

Examples of available exporters:

```text
models/DiffSlack_Trainer.py    -> save_path_data(..., path_data_dir='.../path_data_V2/APF-hard')
models/DC3_Trainer.py          -> save_path_data(..., path_data_dir='.../path_data_V2/DC3-50')
models/ENFORCE_Trainer.py      -> save_path_data(..., path_data_dir='.../path_data_V2/ENFORCE')
models/IL_Trainer.py           -> save_path_data(..., path_data_dir='.../path_data_V2/IL_pure')
models/IL_Hard_Trainer.py      -> save_path_data(...)
```

In each method entry file, uncomment:

```python
trainer.save_path_data()
```

Make sure the output directory matches the method name expected by `carla/track.py`. The tracking script currently supports:

```text
APF-hard
DC3
IL-Soft
IL_pure
DC3-50
ENFORCE
```

The exported files are loaded as:

```text
path_data_V2/<method>/batch_<index>.npy
```

## 5. CARLA Trajectory Tracking

The `carla/` folder contains the trajectory tracking code and a CARLA map asset:

```text
carla/Map.umap
carla/track.py
```

After saving trajectory data with each method's `save_path_data` function, run:

```bash
python carla/track.py
```

At the bottom of `carla/track.py`, choose the method to evaluate:

```python
alg = 'APF-hard'  # 'APF-hard', 'DC3', 'IL-Soft', 'IL_pure', 'DC3-50', 'ENFORCE'
```

The script loads saved trajectories, tracks them in CARLA with PID/feedforward control, and reports tracking metrics such as:

```text
RMSE_CTE (m)
Max_CTE (m)
Avg_Heading_Err (deg)
Control_Smoothness
```

Results are saved under:

```text
res/<method>/
```

## Notes

- Some paths in the current scripts are absolute paths from the original experiment environment. Update them before running on a new machine.
- Checkpoints are organized under `save_dir/`; you can set `load_dir` in each entry script to evaluate a trained model.
- `DiifSlack.py` is the current DiffSlack entry filename in the repository.

## Acknowledgements

This project builds on and benefits from the following open-source projects and tools:

- [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics.git)
- [CARLA](https://github.com/carla-simulator/carla.git)
- [ENFORCE](https://github.com/process-intelligence-research/ENFORCE.git)
- [DC3](https://github.com/locuslab/DC3.git)
