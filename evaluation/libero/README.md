# Evaluation on LIBERO

Four LIBERO suites (`libero_spatial`, `libero_object`, `libero_goal`,
`libero_10`) x 10 tasks x 50 trials = 2000 episodes per checkpoint, reported
as the average success rate. The policy server speaks msgpack over a websocket
and batches concurrent requests; the client runs LIBERO / robosuite.

| | |
| --- | --- |
| Client env | `simvla2-libero-eval` (Python 3.8.13), built by `install_env.sh` |
| Upstream | [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) @ `8f1084e3132a39270c3a13ebe37270a43ece2a01`, unpatched |
| Server defaults | `--dtype bf16 --flow_steps 10`; `run_eval.sh` port 8100 |
| Chunk execution | first 5 actions of each chunk, then replan |

Shared options, result layout and conventions: see `../README.md`.

## Files

| File | Role |
| --- | --- |
| `install_env.sh` | Builds `simvla2-libero-eval` and clones LIBERO into `third_party/LIBERO`. |
| `run_eval.sh` | Evaluates one checkpoint. |
| `serve_simvla2_libero.py` | Policy server (websocket + msgpack, request batching). |
| `libero_client.py` | Runs one suite and writes `<output_dir>/<suite>/results.json`. |

## 1. Environment

The server runs in the main `simvla2` env (top-level `README.md`,
Installation). The client env is separate because LIBERO pins
`robosuite==1.4.0`, `gym==0.25.2` and torch 1.11:

```bash
bash evaluation/libero/install_env.sh            # --env-name NAME, --src-dir DIR
```

The script creates `simvla2-libero-eval`, clones LIBERO at the pinned commit
into `third_party/LIBERO`, installs torch 1.11.0+cu113, LIBERO's
`requirements.txt` and LIBERO itself (editable), then the client's transport
packages (`msgpack-numpy`, `websockets`, `imageio`), and
checks the imports. Re-running it skips what is already in place. For a
checkout elsewhere, pass `--src-dir` here and set `LIBERO_DIR` for
`run_eval.sh`.

## 2. Data

Evaluation needs no download: the BDDL task files and initial states ship
with LIBERO. LIBERO looks them up through `$LIBERO_CONFIG_PATH/config.yaml`
(falling back to `~/.libero/config.yaml`). Unless `LIBERO_CONFIG_PATH` is
already set, `run_eval.sh` uses `<output-base>/.libero/` and, if no
`config.yaml` exists there yet, writes one that points at `LIBERO_DIR` (an
existing one is kept). For a manual client run, create it yourself:

```bash
export LIBERO_DIR=$PWD/third_party/LIBERO
export LIBERO_CONFIG_PATH=$HOME/.simvla2/libero
mkdir -p "$LIBERO_CONFIG_PATH"
cat >"$LIBERO_CONFIG_PATH/config.yaml" <<YAML
benchmark_root: $LIBERO_DIR/libero/libero
bddl_files: $LIBERO_DIR/libero/libero/bddl_files
init_states: $LIBERO_DIR/libero/libero/init_files
datasets: $LIBERO_DIR/libero/libero/datasets
assets: $LIBERO_DIR/libero/libero/assets
YAML
```

The shipped `norm_stats/libero.json` is used for checkpoints that do not carry
their own `norm_stats.json`.

## 3. Run the evaluation

```bash
# One checkpoint, full protocol
bash evaluation/libero/run_eval.sh --ckpt runs/libero/ckpt-200000

# On another GPU
bash evaluation/libero/run_eval.sh --ckpt runs/libero/ckpt-200000 --gpu 1

# Smoke run (not marked _COMPLETE)
bash evaluation/libero/run_eval.sh --ckpt runs/libero/ckpt-200000 \
    --suites libero_spatial --num-trials 5 --no-video
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--suites LIST` | all four | Comma-separated suites. |
| `--num-trials N` | 50 | Trials per task. |
| `--parallel-suites` | off | Run the suites at once against the server instead of one after another. |
| `--no-video` | off | Skip the rollout videos. |
| `LIBERO_DIR` (env) | `third_party/LIBERO` | LIBERO checkout. |

Results of checkpoint `X`:

```
<output-base>/X/
  summary.json             success / total / rate over all episodes, "groups" per suite
  <suite>/results.json     per-suite counts, with per-task "tasks"
  <suite>/videos/          one mp4 per episode (unless --no-video)
  logs/server.log, logs/<suite>.log
```

## 4. Server and client by hand

```bash
# Server (simvla2 env, repo root)
CUDA_VISIBLE_DEVICES=0 python evaluation/libero/serve_simvla2_libero.py \
    --checkpoint runs/libero/ckpt-200000 \
    --norm_stats runs/libero/ckpt-200000/norm_stats.json \
    --port 8000
# Liveness check: the server's first websocket message is its metadata
python -c "from evaluation.policy_client import PolicyClient; print(PolicyClient('127.0.0.1', 8000).metadata)"

# Client (simvla2-libero-eval env, LIBERO_CONFIG_PATH as in section 2)
cd evaluation/libero
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$LIBERO_DIR \
python libero_client.py --port 8000 --task_suite libero_spatial \
    --num_trials 50 --output_dir ./eval_results
```

`run_eval.sh` always passes `--norm_stats` explicitly (the checkpoint's own
file, else the shipped one); do the same by hand so both paths load the stats
identically.

## 5. Interface contract

- **Request** (msgpack, arrays via msgpack-numpy): `observation/image` and
  `observation/wrist_image`, the agent-view and wrist cameras rendered at
  256x256, rotated 180 degrees and resized to 224x224 uint8 (bilinear,
  aspect kept, zero-padded; the server then resizes to the model's image
  size);
  `observation/state`, 8-D `[eef_pos(3), axis_angle(3), gripper_qpos(2)]`;
  `prompt`, the task language.
- **Response**: `{"actions": [H, 7]}`, delta end-effector actions
  `[d_xyz(3), d_rot(3), gripper]` in LIBERO's `OSC_POSE` convention. The client
  sends them to the simulator unchanged.

## 6. Troubleshooting

- **`RuntimeError ... meta device` while loading the checkpoint**: upgrade
  `transformers` in the `simvla2` env to 4.57 or newer (within the
  `requirements.txt` bound).
- **`ImportError: robosuite.environments.manipulation.single_arm_env`**:
  robosuite 1.5 is installed; LIBERO needs `robosuite==1.4.0`. Install it in
  the client env (`pip install robosuite==1.4.0`): `install_env.sh` skips
  LIBERO's requirements while `libero.libero` and `robosuite` both import.
- **`FileNotFoundError` on a BDDL file**: the `config.yaml` in use
  (`$LIBERO_CONFIG_PATH`, else `~/.libero/`) points at another checkout;
  `run_eval.sh` does not rewrite an existing one (section 2).
- **Server stalls at startup on a node without network**: pre-download the
  Hugging Face files and set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.
