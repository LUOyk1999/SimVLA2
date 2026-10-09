# Evaluation

Every benchmark is evaluated the same way: a policy **server** holds one
checkpoint on one GPU (the main `simvla2` env), and the benchmark's simulator
**client** runs in its own conda env and talks to the server over localhost
(msgpack over websockets, `evaluation/serving.py`).

```bash
bash evaluation/libero/install_env.sh                       # once per machine
bash evaluation/libero/run_eval.sh --ckpt runs/libero/ckpt-200000
```

| Benchmark | Client env | Upstream (pinned) | Default protocol | Server dtype / flow steps | Port |
| --- | --- | --- | --- | --- | --- |
| [LIBERO](libero/README.md) | `simvla2-libero-eval` | LIBERO `8f1084e` | 4 suites x 10 tasks x 50 trials | bf16 / 10 | 8100 |
| [WidowX / Bridge](bridge/README.md) | `simvla2-simpler-eval` | SimplerEnv `06accac` + patch | 4 tasks x 24 episodes | bf16 / 30 | 8200 |
| [CALVIN ABC->D](calvin/README.md) | `simvla2-calvin-eval` | CALVIN `fa03f01` | 1000 chains of 5 subtasks | bf16 / 20 | 8300 |
| [Google Robot / Fractal](fractal/README.md) | `simvla2-simpler-eval` | same as WidowX | variant aggregation, 1803 episodes | bf16 / 30 | 8400 |

## Files per benchmark

| File | Role |
| --- | --- |
| `install_env.sh` | Builds the simulator env and clones the official benchmark repo at a pinned commit, plus this repo's patch where one is needed (`evaluation/patches/`). Idempotent. |
| `run_eval.sh` | Evaluates one checkpoint (`--ckpt`). |
| `serve_simvla2_<bench>.py` | Policy server. |
| `<bench>_client.py` | Simulator-side client. |
| `README.md` | Setup, data, protocol, interface contract, troubleshooting. |

## Shared options

Options shared by every `run_eval.sh` (benchmark-specific ones are listed by
`--help` and in each README):

| Option | Meaning |
| --- | --- |
| `--ckpt DIR` | The checkpoint to evaluate (required). |
| `--gpu ID` | GPU for the server and the simulator (default `0`). |
| `--port N` | Server port; default as in the table above. Runs that share a host at the same time need distinct ports. |
| `--output-base DIR` | Results root; default `eval-<name>` next to the directory holding the checkpoint (`runs/libero/ckpt-X` -> `runs/eval-libero/ckpt-X`). |
| `--norm-stats PATH` | Override the normalization stats. By default the checkpoint's own `norm_stats.json` is used, else the shipped `norm_stats/<bench>.json`. |
| `--dtype`, `--flow-steps` | Server weight dtype and flow-matching Euler steps; defaults as in the table above. |

`SERVER_PY` / `CLIENT_PY` override the interpreters (default: the python of
the server env, `simvla2`, and of the client env above, falling back to
`python` when the env does not exist), and `<BENCH>_DIR`
(`LIBERO_DIR`, `SIMPLER_DIR`, `CALVIN_DIR`) points at an
upstream checkout outside the default `third_party/`. On nodes without network
access, pre-download the Hugging Face files (top-level `README.md`,
Installation) and export `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.

## Results layout

Results for checkpoint `X` go to `<output-base>/X/`:

```
summary.json   success / total / rate, plus per-suite or per-task "groups"
               (CALVIN: "score" = mean completed chain length, "sr" = SR@1..5;
               Fractal: "mean_rate" = mean over the three tasks, the reported number)
_COMPLETE      the full protocol finished; rerunning the command skips it
_FAILED        why an attempt failed; the next run retries it
logs/          server.log and the client logs
```

Runs that narrow the protocol (fewer tasks or trials) still write
`summary.json`, but the checkpoint is never marked `_COMPLETE`. Evaluation uses
open-loop chunk execution; how much of each chunk is executed before
replanning is fixed per benchmark and documented in its README.
