# Datasets

Basir is checked against the [TUM RGB-D benchmark](https://cvg.cit.tum.de/data/datasets/rgbd-dataset): handheld Kinect sequences with ground-truth camera poses from a motion-capture system. If a fused room looks wrong, the bug is in Basir's geometry, not in the tracking.

The code is in [`basir/ingest/datasets`](../basir/ingest/datasets/__init__.py) and the command line around it is [`scripts/download_dataset.py`](../scripts/download_dataset.py).

## Download and prepare

```bash
uv run python scripts/download_dataset.py fr1/desk                 # 328 MB
uv run python scripts/download_dataset.py fr1/room                 # 746 MB
uv run python scripts/download_dataset.py --list-presets           # every TUM sequence
uv run python scripts/download_dataset.py fr1/room --name room     # preset, custom folder
uv run python scripts/download_dataset.py https://host/scene.tgz   # any URL
uv run python scripts/download_dataset.py ~/recordings/kitchen.tar.gz   # local archive (or folder)
uv run python scripts/download_dataset.py fr1/room --dry-run       # show source and destination, write nothing
```

One run does the following ([`prepare.py`](../basir/ingest/datasets/prepare.py)):

1. **Resolve** the source: a preset key or alias, a URL, or a local archive / folder.
2. **Download** URLs into `datasets/_archives/`. An interrupted download resumes on the next run and is never mistaken for a finished one. A `--checksum` is verified if given.
3. **Extract** safely (no `../` escapes), flattening the archive's single top-level folder.
4. **Associate**: write `associations.txt`, pairing each RGB frame with the closest unused depth frame within 0.02 s, and `associations_with_gt.txt`, adding the nearest ground-truth pose.
5. **Describe**: write `dataset_meta.json` (name, source, format, frame counts, whether there's ground truth, archive size).

An already prepared folder is left alone (`--overwrite` redoes it); a half-extracted one is detected and extracted again.

| Option | Does |
|---|---|
| `source` | Preset (e.g. `fr1/room`), URL, local archive or folder |
| `-n`, `--name` | Folder name under the output directory |
| `-o`, `--output-dir` | Datasets root (default: `$DATA_DIR/datasets` or `<project>/datasets`) |
| `--no-extract` | Keep the archive only |
| `--remove-archive` | Delete the archive after extraction |
| `-f`, `--overwrite` | Redo an already prepared dataset |
| `--no-flatten` | Keep the archive's single top-level folder |
| `--no-associate` | Skip writing `associations.txt` |
| `--max-difference` | Max RGB-depth time gap in seconds (default 0.02) |
| `--checksum`, `--checksum-type` | Expected digest of the archive (`sha256` or `md5`) |
| `--no-resume` | Restart instead of resuming a partial download |
| `-l`, `--list-presets` | List the presets and exit |
| `--create-toy-dataset` | Create the synthetic dataset (below) |
| `-d`, `--dry-run` | Show what would happen; write nothing |
| `-q`, `--quiet` / `-v`, `--verbose` | Warnings only / debug logging and tracebacks |

Notes:

- **Datasets are big and git-ignored.** They go to `datasets/` (or `$DATA_DIR/datasets`). The two Phase 0 sequences are versioned with DVC instead, see [Data versioning](04-data-versioning.md).
- **A preset that isn't downloaded yet** makes `replay.py`, `visualize_pointcloud.py` and `simulate_phones.py` stop with the exact download command to run. The Temporal pipeline downloads it itself.

## Presets

[`presets.py`](../basir/ingest/datasets/presets.py) has 18 TUM sequences. Each key also has aliases: `fr1/room` resolves from `fr1_room`, `tum_fr1_room`, `freiburg1_room` and `rgbd_dataset_freiburg1_room` (case, `-`, spaces and `/` are normalised).

| Key | Size | Scene |
|---|---|---|
| `fr1/desk` | ~328 MB | Desk with monitor, mugs, books (**Phase 0**) |
| `fr1/room` | ~746 MB | Handheld traversal of a whole office room (**Phase 0**) |
| `fr1/desk2` | ~333 MB | Second desk sequence |
| `fr1/xyz` | ~427 MB | Translation along 3 principal axes |
| `fr1/rpy` | ~391 MB | Rotation around roll, pitch, yaw |
| `fr1/plant` | ~663 MB | House plant on a table |
| `fr1/teddy` | ~838 MB | Teddy bear on an office chair |
| `fr1/360` | ~399 MB | 360 degree turn in the room |
| `fr1/floor` | ~701 MB | Floor loop trajectory |
| `fr2/desk` | ~1.8 GB | Office desk |
| `fr2/xyz` | ~2.1 GB | Translation along 3 axes |
| `fr2/rpy` | ~1.9 GB | Rotation along 3 axes |
| `fr2/large_with_loop` | ~2.7 GB | Industrial hall with loop closure |
| `fr3/long_office` | ~1.4 GB | Long office household loop |
| `fr3/nostructure_texture_near` | ~745 MB | Textured, no structure, near, with loop |
| `fr3/structure_texture_far` | ~497 MB | Structure and texture, far |
| `fr3/sitting_xyz` | ~739 MB | Dynamic scene, people sitting |
| `fr3/walking_xyz` | ~503 MB | Dynamic scene, people walking |

The URLs point at `https://cvg.cit.tum.de/rgbd/dataset/` (TUM moved the benchmark there; the old host only redirects). `uv run pytest -m network` checks that every preset URL still answers.

## The TUM format

A prepared folder looks like this. Anything saved in the same layout, such as a future phone recording, works with every script.

```text
rgbd_dataset_freiburg1_desk/
├── rgb/<timestamp>.png              # colour, 640x480
├── depth/<timestamp>.png            # 16-bit depth, metres x 5000
├── rgb.txt                          # timestamp filename
├── depth.txt                        # timestamp filename
├── groundtruth.txt                  # timestamp tx ty tz qx qy qz qw   (T_world_cam)
├── associations.txt                 # written by Basir: t_rgb rgb t_depth depth
├── associations_with_gt.txt         # written by Basir: ... + t_gt tx ty tz qx qy qz qw
├── dataset_meta.json                # written by Basir
└── intrinsics.json                  # optional: overrides the sensor calibration
```

[`tum_format.py`](../basir/ingest/datasets/tum_format.py) parses and writes the timestamped text files (standard library only) and holds the per-sensor calibration listed in [Geometry](02-geometry.md#camera-intrinsics).

## Reading frames

[`TUMDatasetSource`](../basir/ingest/sources/tum.py) reads `associations_with_gt.txt` (or `associations.txt` without poses) and yields `Frame`s: RGB, depth in metres, intrinsics and the ground-truth pose. Options: `stride` (every Nth frame), `max_frames`, and `require_pose` (default on; frames without a pose are skipped). `intrinsics_source` says where the camera came from.

It implements the [`FrameSource`](../basir/ingest/sources/base.py) interface (`intrinsics`, `__iter__`, `__len__`, `close`), which phone recordings and the live ARCore stream will also implement, so later phases don't care where frames come from.

## The toy dataset

```bash
uv run python scripts/download_dataset.py --create-toy-dataset --name toy_room
```

[`toy.py`](../basir/ingest/datasets/toy.py) writes a tiny synthetic TUM-format scene with known geometry: 5 frames of 64x48 at ~30 fps, frame *i* at a constant depth of 1.5 m + 0.05 m x *i* (stored with the factor 5000), depth timestamps 2.1 ms after RGB like a real sensor, a simple ground-truth trajectory, and an `intrinsics.json` with the default camera scaled to 64x48. It's what the offline tests use, and it's handy for trying the scripts without a download.
