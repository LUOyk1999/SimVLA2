# SimVLA 2.0: Instruction-Conditioned Control without Language Pretraining

```
observation                       instruction
    |                                  |
SigLIP2 vision encoder          fixed tokenizer -> discrete codes
(pretrained, vision-only)       random-init embedding (action-trained)
    |                                  |
    +---------------+------------------+
                    |
        Joint action Transformer  (+ proprioception, flow time)
                    |
        conditional flow-matching head
                    |
              action chunk [H, da]
```

## Results

| Benchmark | Metric | SimVLA 2.0 |
| --- | --- | --- |
| LIBERO | Avg. success | **96.9%** |
| CALVIN ABC→D | Avg. completed chain length | **4.24 / 5** |
| WidowX | Avg. success | **91.7%** |
| Google Robot | Avg. success | **82.0%** |

Parameters: SigLIP2 vision ≈ 303M, action-trained instruction embedding ≈ 31M,
action Transformer ≈ 303M, **pretrained language-side parameters: 0**
(≈ 639M total).

## Installation

```bash
conda create -y -n simvla2 python=3.11 && conda activate simvla2
pip install "torch>=2.6" "torchvision>=0.21" --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
# Bridge / Fractal training only (RLDS readers):
pip install "tensorflow-cpu>=2.15,<3.0" tensorflow_datasets
```

The evaluation scripts expect the env at `<conda>/envs/simvla2`. The vision
tower (`google/siglip2-large-patch16-256`) and tokenizer are downloaded on
first use. Only the **tokenizer vocabulary** of `bert-base-uncased` is used; no
BERT weights are loaded. For offline nodes, pre-download both and export
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`:

```bash
huggingface-cli download google/siglip2-large-patch16-256
huggingface-cli download bert-base-uncased --include "*.json" "*.txt"
```

## Data

`datasets/metas/` ships one episode meta per benchmark (plus the Bridge
instruction-paraphrase map `bridge_lang.json`). Replace the data-root
placeholder with your own path, e.g.

```bash
sed -i "s|<LIBERO_ROOT>|/path/to/libero|g" datasets/metas/libero.json
```

## Training

```bash
bash training/train_libero.sh   4 0,1,2,3   # [NUM_GPUS] [GPU_IDS]
bash training/train_calvin.sh   4 0,1,2,3
bash training/train_bridge.sh   4 0,1,2,3   # WidowX / Bridge V2
bash training/train_fractal.sh  4 0,1,2,3   # Google Robot / Fractal
```

| Domain | Views | Action dim | Chunk H | Global batch | Base LR | Vision LR × | Iterations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LIBERO | 2 | 7 | 10 | 256 | 5e-4 | 0.10 | 200k |
| CALVIN | 2 | 20 | 30 | 128 | 1e-4 | 0.10 | 200k |
| WidowX | 1 | 7 | 30 | 256 | 1e-4 | 0.05 | 100k |
| Google Robot | 1 | 7 | 30 | 256 | 1e-4 | 0.05 | 200k |

## Evaluation

```bash
bash evaluation/libero/install_env.sh                       # once per machine
bash evaluation/libero/run_eval.sh --ckpt runs/libero/ckpt-200000
```

The same two steps work for `calvin`, `bridge` and `fractal`. Each benchmark
runs the policy server in the `simvla2` env and the simulator client in its own
env. Shared options and the results layout are in
[`evaluation/README.md`](evaluation/README.md); protocols and troubleshooting
are in [LIBERO](evaluation/libero/README.md),
[CALVIN](evaluation/calvin/README.md),
[WidowX](evaluation/bridge/README.md) and
[Google Robot](evaluation/fractal/README.md).
