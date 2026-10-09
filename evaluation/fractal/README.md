# Evaluation on SimplerEnv: Google Robot / Fractal (RT-1)

The Google-robot variant-aggregation (VA) track of SimplerEnv: `coke_can`
(33 scenarios, 825 episodes), `move_near` (10 scenarios, 600 episodes) and
`open_close` (42 scenarios, 378 episodes), 1803 episodes per checkpoint. The
policy server speaks msgpack over a websocket and batches concurrent requests;
the client runs SAPIEN. The scenario definitions live in
`configs/VA/<task>.json`.

| | |
| --- | --- |
| Client env | `simvla2-simpler-eval`, the same env as WidowX / Bridge |
| Upstream | [SimplerEnv](https://github.com/simpler-env/SimplerEnv) @ `06accaca93535902d408da4855f21cece12bceb7` + `evaluation/patches/simplerenv.patch` |
| Server defaults | `--dtype bf16 --flow_steps 30`; `run_eval.sh` port 8400 |
| Chunk execution | first 20 actions of each chunk, then replan |

Shared options, result layout and conventions: see `../README.md`.

## Files

| File | Role |
| --- | --- |
| `install_env.sh` | Same as `../bridge/install_env.sh` (it runs that script). |
| `run_eval.sh` | Evaluates one checkpoint. |
| `serve_simvla2_fractal.py` | Policy server (websocket + msgpack, request batching). |
| `fractal_client.py` | Runs a range of scenarios of one task against the server. |
| `configs/VA/` | The three VA task definitions. |

## 1. Environment

The server runs in the main `simvla2` env. The client shares
`simvla2-simpler-eval` and the patched SimplerEnv checkout with Bridge; if you
built them for Bridge there is nothing to do, otherwise:

```bash
bash evaluation/fractal/install_env.sh           # --env-name NAME, --src-dir DIR
```

See `../bridge/README.md` (section 1) for what the script and the patch do and
for the Vulkan setup. Fractal uses the patch's absolute-pose controller for the
Google robot, `arm_pd_ee_base_pose_align_interpolate_by_planner`.

## 2. Data

Nothing to stage: SimplerEnv ships the scenes and overlays. The shipped
`norm_stats/fractal.json` matches the training statistics and is used for
checkpoints without their own `norm_stats.json`.

## 3. Run the evaluation

```bash
bash evaluation/fractal/run_eval.sh --ckpt runs/fractal/ckpt-200000
bash evaluation/fractal/run_eval.sh --ckpt runs/fractal/ckpt-200000 --gpu 0 --render-gpu 1

# Smoke run: move_near, two episodes per scenario (not marked _COMPLETE)
bash evaluation/fractal/run_eval.sh --ckpt runs/fractal/ckpt-200000 --tasks move_near --episode-ids 0,1
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--tasks LIST` | `coke_can,move_near,open_close` | Comma-separated tasks. |
| `--workers N` | 4 | Simulator processes per task; the scenarios are split between them. The tasks run one after another. |
| `--episode-ids LIST` | all | Only these episode ids within each `move_near` scenario (`coke_can` varies object positions and `open_close` the robot's initial position instead; both ignore it). |
| `--gripper-threshold F` | 0.5 | Gripper open/close threshold. |
| `--render-gpu ID` | the `--gpu` GPU | Render on a second card. |
| `SIMPLER_DIR` (env) | `third_party/SimplerEnv` | Patched SimplerEnv checkout. |

The client always runs with `--resume`, so re-running continues a
half-finished checkpoint.

Results of checkpoint `X`:

```
<output-base>/X/
  summary.json                       "groups" per task; "rate" pools all 1803
                                     episodes, "mean_rate" averages the three
                                     task rates (the reported number)
  VA/<task>/google_results.txt       one JSON line per episode
  VA/<task>/<scenario>_<ep>_<succ>.mp4
  logs/server.log, logs/<task>-worker<w>.log
```

## 4. Server and client by hand

```bash
# Server (simvla2 env, repo root)
CUDA_VISIBLE_DEVICES=0 python evaluation/fractal/serve_simvla2_fractal.py \
    --checkpoint runs/fractal/ckpt-200000 \
    --norm_stats runs/fractal/ckpt-200000/norm_stats.json \
    --port 8000
# Liveness check: the server's first websocket message is its metadata
python -c "from evaluation.policy_client import PolicyClient; print(PolicyClient('127.0.0.1', 8000).metadata)"

# Client (simvla2-simpler-eval env)
export SIMPLER_DIR=$PWD/third_party/SimplerEnv
cd evaluation/fractal
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$SIMPLER_DIR:$SIMPLER_DIR/ManiSkill2_real2sim \
python fractal_client.py --port 8000 --task coke_can --output_dir ./eval_results \
    --scenario_start 0 --scenario_end 33 --resume
```

`run_eval.sh` always passes `--norm_stats` explicitly (the checkpoint's own
file, else the shipped one); do the same by hand. `--scenario_start` /
`--scenario_end` index the entries of `configs/VA/<task>.json`.

## 5. Interface contract

- **Request** (msgpack, arrays via msgpack-numpy): `image`, the overhead
  camera frame; `proprio`, 7-D, all zeros (as in training); `language_instruction`.
- **Response**: `{"actions": [H, 7]}`, `[delta_xyz(3), euler_xyz(3), gripper]`:
  position offsets in meters in the robot base frame, an absolute Euler
  orientation target, and a continuous gripper value that the server clips to
  `[-1, 1]`. A failed request gets a text frame instead; the client raises and
  the worker exits, so the checkpoint is marked `_FAILED`.
- **Client side**: the position offsets of the chunk are added to an anchor
  position, giving absolute targets. The anchor is the measured end-effector
  position for an episode's first chunk and the last commanded target after
  that. The orientation is passed through; the gripper becomes +1 above
  `--gripper_threshold` and -1 otherwise. The absolute pose goes to the
  patched `arm_pd_ee_base_pose_align_interpolate_by_planner` controller.

## 6. Troubleshooting

- **`AssertionError: Task ... is not supported`**: SimplerEnv is not the
  patched checkout. The VA configs use raw ManiSkill env ids
  (`GraspSingleOpenedCokeCanInScene-v0`, ...), which stock `simpler_env.make`
  rejects; the patch falls back to `gym.make` for them. Check `SIMPLER_DIR`
  (see `../bridge/README.md`, Troubleshooting).
- **A worker hangs at "creating engine" with the GPU idle**: SAPIEN's Vulkan
  setup deadlocked. Use a node with a working Vulkan ICD
  (`VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`). The run waits on the
  hung worker: kill it and re-run (`--resume` keeps the finished episodes).
