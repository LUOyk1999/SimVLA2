# Evaluation on CALVIN (ABC -> D)

CALVIN's long-horizon protocol on the D environment: chains of 5 language
subtasks, scored as the mean number of completed subtasks per chain ("score")
plus the success rate at chain lengths 1-5 ("sr"). The policy server speaks
msgpack over a websocket and batches concurrent requests; the client drives
CALVIN's PyBullet env.

| | |
| --- | --- |
| Client env | `simvla2-calvin-eval` (Python 3.8), built by `install_env.sh` |
| Upstream | [CALVIN](https://github.com/mees/calvin) @ `fa03f01f19c65920e18cf37398a9ce859274af76` (submodules calvin_env @ `1431a46`, tacto @ `dd53360`), unpatched |
| Server defaults | `--dtype bf16 --flow_steps 20`; `run_eval.sh` port 8300 |
| Chunk execution | first 30 actions of each chunk (`--plan-len`), then replan |

Shared options, result layout and conventions: see `../README.md`.

## Files

| File | Role |
| --- | --- |
| `install_env.sh` | Builds `simvla2-calvin-eval` and clones CALVIN into `third_party/calvin`. |
| `run_eval.sh` | Evaluates one checkpoint. |
| `serve_simvla2_calvin.py` | Policy server (websocket + msgpack, request batching). |
| `calvin_client.py` | Runs the chains, in parallel worker processes, and writes `summary.json`. |
| `ABC_D/validation/` | CALVIN's task oracle and validation annotations, loaded by the client. |

## 1. Environment

The server runs in the main `simvla2` env. CALVIN pins torch 1.13,
pytorch-lightning 1.8.6 and gym 0.21, so the client gets its own env:

```bash
bash evaluation/calvin/install_env.sh            # --env-name NAME, --src-dir DIR
```

CALVIN's own `install.sh` does not work on current hosts: `gym==0.21` only
builds with pip < 24 and setuptools < 58, and `calvin_models` declares an
unpinned dep list that replaces torch 1.13. The script therefore

1. creates `simvla2-calvin-eval` and clones CALVIN at the pinned commit with
   its submodules into `third_party/calvin`;
2. pins the build tooling (pip 23.3.2, setuptools 57.5.0, wheel 0.38.4, cmake
   3.27.9);
3. installs torch 1.13.1 + torchvision 0.14.1 (cu117), then pytorch-lightning
   1.8.6 and gym 0.21 together on top of it;
4. installs tacto, `calvin_env` and, with `--no-deps`, `calvin_models`;
5. adds the runtime packages `calvin_models` imports at evaluation time
   (hydra-core 1.1.1, omegaconf 2.1.2, `numpy<1.24`, opencv, pyhash, ...) and the
   client's `msgpack-numpy` / `websockets`, then checks the imports.

Re-running it skips what is already in place. If a failed attempt left broken
`gym` metadata in pip's cache, run `pip cache purge` before retrying. For a
checkout elsewhere, pass `--src-dir` here and set `CALVIN_DIR` for
`run_eval.sh`.

## 2. Data

The evaluation runs on the validation split of `task_ABC_D`. CALVIN's
downloader fetches and unpacks it in place:

```bash
cd third_party/calvin/dataset
sh download_data.sh ABC      # task_ABC_D, ~517 GB (training + validation in one archive)
sh download_data.sh debug    # calvin_debug_dataset, ~1.3 GB, for smoke tests only
```

`run_eval.sh` reads `$CALVIN_DIR/dataset/task_ABC_D/validation` unless
`--dataset-path` says otherwise.

**Fix the baked-in asset path.** Each split carries a
`.hydra/merged_config.yaml` whose `data_path` is the absolute path of
`calvin_env/data` (the scene URDFs, not the dataset) on the machine that
assembled the release; `get_env()` loads the scene from it and fails with
`Cannot load URDF file`. Point it at your checkout's `calvin_env/data`:

```bash
ROOT=$PWD/third_party/calvin/dataset/task_ABC_D
grep -n data_path $ROOT/validation/.hydra/merged_config.yaml   # note the baked-in path
BAKED=<the path printed above>
sed -i "s|$BAKED|$PWD/third_party/calvin/calvin_env/data|g" \
    $ROOT/validation/.hydra/merged_config.yaml $ROOT/training/.hydra/merged_config.yaml
```

Do the same for `calvin_debug_dataset` if you use it.

The shipped `norm_stats/calvin.json` is used for checkpoints without their own
`norm_stats.json`.

## 3. Run the evaluation

```bash
bash evaluation/calvin/run_eval.sh --ckpt runs/calvin/ckpt-200000
bash evaluation/calvin/run_eval.sh --ckpt runs/calvin/ckpt-200000 --gpu 1

# Quick check on 100 chains (not marked _COMPLETE)
bash evaluation/calvin/run_eval.sh --ckpt runs/calvin/ckpt-200000 --num-sequences 100
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--dataset-path DIR` | `$CALVIN_DIR/dataset/task_ABC_D/validation` | Validation split. |
| `--num-sequences N` | 1000 | Evaluation chains; fewer is a partial run. |
| `--num-workers N` | 5 | CALVIN env processes (PyBullet is CPU-bound). |
| `--plan-len N` | 30 | Actions executed per server call. |
| `--gripper-threshold F` | 0.0 | The gripper opens above this value. |
| `--video` | off | Save one mp4 per subtask (its last attempt) as `<instruction>_<True\|False>.mp4` in the worker directory; later chains overwrite files with the same name. |
| `CALVIN_DIR` (env) | `third_party/calvin` | CALVIN checkout. |

The client renders with EGL on the `--gpu` GPU (`EGL_VISIBLE_DEVICES`,
`PYOPENGL_PLATFORM=egl`). As with LIBERO, a CALVIN run is not resumable
halfway: an interrupted checkpoint starts over.

Results of checkpoint `X`:

```
<output-base>/X/
  summary.json     {"score": mean chain length, "sr": [SR@1, ..., SR@5],
                    "num_sequences", "num_workers", "eval_start", "eval_end"}
  worker_<w>_s<start>_e<end>/
    log.txt        running SR after every chain in that worker's range
                   (with --num_workers 1, X/log.txt instead)
  logs/server.log, logs/client.log
```

## 4. Server and client by hand

```bash
# Server (simvla2 env, repo root)
CUDA_VISIBLE_DEVICES=0 python evaluation/calvin/serve_simvla2_calvin.py \
    --checkpoint runs/calvin/ckpt-200000 \
    --norm_stats runs/calvin/ckpt-200000/norm_stats.json \
    --port 8000
# Liveness check: the server's first websocket message is its metadata
python -c "from evaluation.policy_client import PolicyClient; print(PolicyClient('127.0.0.1', 8000).metadata)"

# Client (simvla2-calvin-eval env)
export CALVIN_DIR=$PWD/third_party/calvin
cd evaluation/calvin
CUDA_VISIBLE_DEVICES=0 EGL_VISIBLE_DEVICES=0 PYOPENGL_PLATFORM=egl \
PYTHONPATH=$CALVIN_DIR/calvin_models:$CALVIN_DIR/calvin_env:$PWD/../.. \
python calvin_client.py --port 8000 --num_workers 5 --num_sequences 1000 \
    --dataset_path $CALVIN_DIR/dataset/task_ABC_D/validation --output_dir ./eval_results --no_video
```

`run_eval.sh` always passes `--norm_stats` explicitly (the checkpoint's own
file, else the shipped one); do the same by hand. For a quick check, use
`--num_sequences 10 --num_workers 1`; `--eval_start` / `--eval_end` select a
range of chains.

## 5. Interface contract

- **Request** (msgpack, arrays via msgpack-numpy): `image0`, the static
  camera; `image1`, the gripper camera; `proprio`, 20-D float64, i.e. the 10-D
  arm state `[xyz(3), rot6d(6), gripper(1)]` followed by 10 zeros;
  `language_instruction`, the subtask annotation. Flow steps are a server
  setting.
- **Response**: `{"actions": [H, 20]}`. The first 10 dims are an absolute
  end-effector target `[xyz(3), rot6d(6), gripper(1)]`. A failed request gets
  a text frame instead; the client raises, and the run exits nonzero without
  writing `summary.json`.
- **Client side**: rot6d is converted to a quaternion, and the gripper opens
  (+1) above `--gripper_threshold` and closes (-1) otherwise; position,
  orientation and gripper go to CALVIN's absolute action interface. Each
  subtask starts from the measured state; within a subtask, the first 10 dims
  of the last executed prediction are sent back as proprio.

## 6. Troubleshooting

- **`Cannot load URDF file` on env reset**: the dataset's
  `merged_config.yaml` `data_path` does not point at your `calvin_env/data`
  (section 2).
- **`FileNotFoundError` on `ABC_D/validation/new_playtable_*.yaml`**: the
  client loads them from its own directory; keep `calvin_client.py` next to
  `ABC_D/`.
- **PyBullet renders black or the GPU stays idle**: EGL is not set up; set
  `EGL_VISIBLE_DEVICES` to the same GPU as `CUDA_VISIBLE_DEVICES` and
  `PYOPENGL_PLATFORM=egl`.
- **libstdc++ ABI errors when CALVIN imports**: preload the env's copy,
  `LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6`.
- **Do not merge the server and client envs**: CALVIN's torch / Lightning /
  gym pins break the model's dependencies.
