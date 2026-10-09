# Evaluation on SimplerEnv: WidowX / Bridge V2

The WidowX visual-matching track of SimplerEnv: 4 tasks
(`widowx_stack_cube`, `widowx_put_eggplant_in_basket`,
`widowx_carrot_on_plate`, `widowx_spoon_on_towel`) x 24 episodes = 96 episodes
per checkpoint, reported as the average success rate. The policy server
speaks msgpack over a websocket and batches concurrent requests; the client
runs SAPIEN.

| | |
| --- | --- |
| Client env | `simvla2-simpler-eval` (Python 3.10, shared with Fractal), built by `install_env.sh` |
| Upstream | [SimplerEnv](https://github.com/simpler-env/SimplerEnv) @ `06accaca93535902d408da4855f21cece12bceb7` (submodule ManiSkill2_real2sim @ `ef7a4d4`) + `evaluation/patches/simplerenv.patch` |
| Server defaults | `--dtype bf16 --flow_steps 30`; `run_eval.sh` port 8200 |
| Chunk execution | the whole predicted chunk, then replan |

Shared options, result layout and conventions: see `../README.md`.

## Files

| File | Role |
| --- | --- |
| `install_env.sh` | Builds `simvla2-simpler-eval` and clones + patches SimplerEnv into `third_party/SimplerEnv` (also used by Fractal). |
| `run_eval.sh` | Evaluates one checkpoint. |
| `serve_simvla2_bridge.py` | Policy server (websocket + msgpack, request batching). |
| `bridge_client.py` | Runs episodes of one task (or `--task all`) against the server. |

## 1. Environment

The server runs in the main `simvla2` env (top-level `README.md`,
Installation). The client env has no torch; it needs SAPIEN 2.x, numpy<2 and
`setuptools<70`, which is why it is separate:

```bash
bash evaluation/bridge/install_env.sh            # --env-name NAME, --src-dir DIR
```

The script creates `simvla2-simpler-eval`, clones SimplerEnv at the pinned
commit (with its submodule) into `third_party/SimplerEnv`, applies the patch,
installs `setuptools<70`, `numpy==1.24.4`, ManiSkill2_real2sim and SimplerEnv
(editable) and the client packages (`msgpack-numpy`, `websockets`, `scipy`,
`mediapy`, ...), and checks the imports. Re-running it skips what is already
in place. For a checkout elsewhere, pass `--src-dir` here and set `SIMPLER_DIR`
for `run_eval.sh`.

**Why the patch.** SimVLA2 predicts absolute end-effector poses, but stock
SimplerEnv drives the WidowX with a delta controller; feeding it absolute
poses gives near-zero success. The patch

- registers an absolute, base-frame pose controller for the WidowX
  (`arm_pd_ee_target_base_pose`) and makes it the default control mode in
  `put_on_in_scene.py` and `env_builder.py`;
- registers the matching controller for the Google robot
  (`arm_pd_ee_base_pose_align_interpolate_by_planner`, used by Fractal);
- reads the rotation of absolute pose targets as Euler xyz instead of
  axis-angle;
- sets the sink scene's (`widowx_put_eggplant_in_basket`) third-person camera
  and robot initial pose to the Bridge-dataset ones;
- adds the TCP pose to the drawer scenes' observations and lets
  `simpler_env.make()` build a registered env id with explicit kwargs; the
  Fractal client relies on both;
- turns off the OptiX denoiser of the ray-traced shader, which only the
  Google-robot drawer scenes use; on driver 580 it renders black frames.

**Vulkan.** SAPIEN renders through Vulkan and needs a loader that sees the
NVIDIA driver:

```bash
sudo apt-get install -y libvulkan1 vulkan-tools
# If `vulkaninfo` cannot find the GPU, register the NVIDIA ICD:
sudo mkdir -p /etc/vulkan/icd.d
sudo tee /etc/vulkan/icd.d/nvidia_icd.json >/dev/null <<'JSON'
{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.0"}}
JSON
```

The client defaults `VK_ICD_FILENAMES` to that file; set it if your ICD lives
elsewhere.

## 2. Data

Nothing to stage: SimplerEnv ships the scenes and real-image overlays. The
shipped `norm_stats/bridge.json` (computed on Bridge V2, identical to the
training stats) is used for checkpoints without their own `norm_stats.json`.
Do not recompute it on a subset: the server unnormalizes absolute poses with
its `state` statistics.

## 3. Run the evaluation

```bash
bash evaluation/bridge/run_eval.sh --ckpt runs/bridge/ckpt-100000
bash evaluation/bridge/run_eval.sh --ckpt runs/bridge/ckpt-100000 --gpu 1
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--tasks LIST` | all four | Comma-separated tasks. |
| `--episodes N` | 24 | Episodes per task. |
| `--workers N` | 8 | Simulator processes per task, rendering on the server's GPU. SAPIEN is CPU-bound at this size; keep it at 8 or below. |
| `SIMPLER_DIR` (env) | `third_party/SimplerEnv` | Patched SimplerEnv checkout. |

Each worker gets its own shader cache, and each render GPU its own
`XDG_RUNTIME_DIR`, so parallel SAPIEN processes do not fight over them. The
client always runs with `--resume`, so re-running continues a half-finished
checkpoint.

Results of checkpoint `X`:

```
<output-base>/X/
  summary.json                   success / total / rate, "groups" per task
  <task>/<task>_results.txt      one JSON line per episode
  <task>/<task>_<ep>_<succ>.mp4  one video per episode
  logs/server.log, logs/<task>-worker<w>.log
```

## 4. Server and client by hand

```bash
# Server (simvla2 env, repo root)
CUDA_VISIBLE_DEVICES=0 python evaluation/bridge/serve_simvla2_bridge.py \
    --checkpoint runs/bridge/ckpt-100000 \
    --norm_stats runs/bridge/ckpt-100000/norm_stats.json \
    --port 8000
# Liveness check: the server's first websocket message is its metadata
python -c "from evaluation.policy_client import PolicyClient; print(PolicyClient('127.0.0.1', 8000).metadata)"

# Client (simvla2-simpler-eval env)
export SIMPLER_DIR=$PWD/third_party/SimplerEnv
cd evaluation/bridge
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$SIMPLER_DIR:$SIMPLER_DIR/ManiSkill2_real2sim \
python bridge_client.py --port 8000 --task all --output_dir ./eval_results --resume
```

`run_eval.sh` always passes `--norm_stats` explicitly (the checkpoint's own
file, else the shipped one); do the same by hand. `--episode_start` /
`--episode_end` select a range of episodes, which is how the launcher splits a
task between workers.

## 5. Interface contract

- **Request** (msgpack, arrays via msgpack-numpy): `image`, the third-person
  camera frame; `proprio`, 7-D `[x, y, z, rx, ry, rz, gripper]`;
  `language_instruction`. Flow steps are a server setting.
- **Response**: `{"actions": [H, 7]}`, absolute end-effector poses in the
  WidowX base frame, `[x, y, z, euler_xyz(3), gripper]` with the gripper in
  `[0, 1]`. The server unnormalizes them with the `state` statistics. A failed
  request gets a text frame instead; the client raises, so the episode is not
  recorded and the checkpoint is not marked `_COMPLETE`.
- **Client side**: position is clipped to +-1 m; the rotation gets the
  `[0, pi/2, 0]` axis offset between the dataset and the simulator frames and
  is clipped to +-pi/2; the gripper becomes close (-1) below 0.91 and open (+1)
  otherwise. The resulting absolute pose goes to the patched
  `arm_pd_ee_target_base_pose` controller.
- **Proprio**: the first request sends the simulator's end-effector pose; after
  that, the last executed prediction is sent back as proprio.

## 6. Troubleshooting

- **Success near 0% and episodes run to the step limit**: the simulator is using
  the stock delta controller. Check that `SIMPLER_DIR` points at the patched
  checkout (`git -C $SIMPLER_DIR/ManiSkill2_real2sim diff --stat` should list
  the patched files) and that no other `simpler_env` is installed in the env.
- **`argument --task: invalid choice`**: the task lists of Bridge and Fractal
  differ. Bridge takes its four `widowx_*` tasks (or `all`); Fractal takes
  `coke_can`, `move_near` and `open_close`. Do not copy task lists between the
  two.
- **`no vulkan device` / SAPIEN fails to create the renderer**: install the
  Vulkan loader and ICD (section 1) and set
  `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`.
- **Comparing dtypes**: numbers are only comparable between runs with the same
  `--dtype` (bf16 by default).
