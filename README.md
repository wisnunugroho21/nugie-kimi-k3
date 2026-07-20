# Kimi K3 recreation — a hybrid linear-attention code LM in JAX

A from-scratch recreation of the [**Kimi K3** architecture](https://www.kimi.com/blog/kimi-k3)
(Moonshot AI, July 2026)
in JAX / Flax NNX, trained as a code-autocomplete LM on CodeParrot. K3's backbone
is the hybrid linear-attention transformer of *Kimi Linear*, extended with
depth-wise attention residuals, gated MLA, and a latent-space MoE; until the K3
technical report lands, each feature here is implemented from its best public
source (annotated throughout the code).

## Architecture

Decoder-only LM: `Embed → [DecoderLayer] × n_layers → RMSNorm → LM head`, where
3 of every 4 layers use a linear-attention token mixer and the 4th uses full
softmax attention (Kimi Linear's 3:1 hybrid), all threaded through a
depth-attention residual backbone.

| Feature | Paper | Where | Notes |
| --- | --- | --- | --- |
| **Block Attention Residuals** | [arXiv:2603.15031](https://arxiv.org/abs/2603.15031) | `kimi_k3_gdn2.py` | Softmax attention over *depth*: each sub-layer's input is a learned mixture of the embedding, completed block sums, and the current partial sum. `attn_res: false` restores the plain pre-norm residual stream. |
| **Gated DeltaNet-2** | [arXiv:2605.22791](https://arxiv.org/abs/2605.22791) | `gated_deltanet_2/` | The linear-attention mixer (¾ of layers). Deliberate stand-in for K3's Kimi Delta Attention — same gated-delta-rule family, but with decoupled erase (`b`) and write (`w`) gates. Five interchangeable chunkwise cores with different numerical/performance trade-offs (`gdn_core`: faithful / stacked_rhs / centered / subchunking / pairwise). |
| **Gated NoPE MLA** | [arXiv:2505.06708](https://arxiv.org/abs/2505.06708) (gate) | `multi_latent_attention/attention.py` | The full-attention mixer (¼ of layers), in absorbed form — one latent serves as both K and V, so the decode cache stores only latents. No positional encoding (the recurrent layers carry position). Head-wise sigmoid output gate (K3's "Gated MLA"). |
| **LatentMoE** | [arXiv:2601.18089](https://arxiv.org/abs/2601.18089) | `multi_latent_attention/moe.py` | Every layer's channel mixer: routed experts run in a shared low-rank latent (α = d_model/d_latent = 4 in the main configs), with a full-width shared expert, sigmoid routing, group-limited routing, and DeepSeek-V3 aux-loss-free bias balancing plus a small sequence-level balancing loss. |
| **Muon + Adam fallback** | [arXiv:2502.16982](https://arxiv.org/abs/2502.16982) (Moonlight) | `pipeline/optimizer.py` | Hidden weight matrices → Muon with consistent-RMS scaling; embedding, LM head, biases, norms, and decay parameters → Optax's AdamW fallback with weight decay disabled. Weight decay is applied only to Muon-side matrices. |

Not yet recreated (awaiting the K3 technical report): the "Stable" LatentMoE
additions, Quantile Balancing, Per-Head Muon, and K3's multimodal / 1M-context
/ MXFP4 machinery.

## Repository layout

```
kimi_k3_gdn2.py            The model: config, AttnRes backbone, DecoderLayer, KimiK3
gated_deltanet_2/
  core.py                  Gated Delta Rule-2 recurrence: recurrent oracle + 5 chunkwise cores
  layer.py                 The GDN-2 token-mixer layer (projections, convs, gates, GQA folding)
multi_latent_attention/
  attention.py             Gated NoPE MLA (absorbed form) + streaming cache
  moe.py                   LatentMoE — routed experts in the shared latent
pipeline/
  config.py                YAML → typed ExperimentConfig
  prepare_data.py          Stage 1: tokenize CodeParrot (or synthetic) into packed memmaps
  data.py                  Stage 2: Grain loader over the memmaps
  train.py                 Stage 3: training loop (data-parallel via shard_map)
  optimizer.py             Muon/AdamW split
  evaluate.py              Stage 4: val loss/ppl + autoregressive generation
  checkpointing.py         Orbax composite checkpoints (model/optimizer/rngs/data iterator)
configs/                   Ready-made experiment YAMLs (see table below)
tests/                     Numerical verification suite (pytest)
```

## Reproducible environment

Python 3.12.13 and every direct/transitive dependency are captured by
`pyproject.toml`, `.python-version`, and the checked-in universal `uv.lock`.
Install [uv 0.11.29](https://docs.astral.sh/uv/getting-started/installation/)
(the version pinned in CI), then create the exact CPU development/test environment:

```bash
uv sync --frozen
uv run --frozen pytest -m "not multi_device"
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen pyright
```

`requirements.txt` remains an unpinned compatibility input for pip users; it is
not the reproducible installation path.

### NVIDIA CUDA environment

JAX's prebuilt NVIDIA wheels are Linux-only (native Windows GPU is unsupported;
WSL2 support is experimental). The lock contains mutually exclusive CUDA 12 and
CUDA 13 overlays, while the default sync remains CPU-only:

```bash
# Recommended for a sufficiently recent Linux driver (CUDA 13 requires >= 580):
uv sync --frozen --extra cuda13

# Older supported Linux drivers / hosted environments (CUDA 12 requires >= 525):
uv sync --frozen --extra cuda12

# Repeat the selected extra when uv checks/runs the environment:
uv run --frozen --extra cuda13 python -c "import jax; print(jax.devices())"
```

Do not enable both CUDA extras together. uv rejects that combination explicitly.

## Quickstart

Fully offline smoke run (synthetic bytes, laptop CPU, minutes):

```bash
python -m pipeline.prepare_data --config configs/tiny.yaml
python -m pipeline.train        --config configs/tiny.yaml
python -m pipeline.evaluate     --config configs/tiny.yaml --eval
python -m pipeline.evaluate     --config configs/tiny.yaml --generate
```

Real training (streams CodeParrot from the HF Hub, BPE tokenizer, bf16):

```bash
python -m pipeline.prepare_data --config configs/base.yaml
python -m pipeline.train        --config configs/base.yaml           # --resume to continue
python -m pipeline.evaluate     --config configs/base.yaml --generate \
    --prompt "def quicksort(arr):" --max-new-tokens 128 --temperature 0.8 --top-p 0.95
```

## Configs

| Config | Params | Target hardware | Notes |
| --- | --- | --- | --- |
| `tiny.yaml` | 2.5M | laptop CPU | Synthetic data, byte vocab — end-to-end smoke test |
| `colab_t4.yaml` | 98M | 1× T4 (Colab) | bf16 for memory (T4 has no bf16 tensor cores) |
| `kaggle_2xt4.yaml` | 98M | 2× T4 (Kaggle) | Same model, data-parallel across both GPUs |
| `base.yaml` | 148M | one modern GPU | The reference single-GPU recipe |
| `h200.yaml` | 1.1B | H200 141 GB | Sparse-MoE run with headroom to scale further |

Configuration ranges and cross-field constraints are validated at startup. In
particular, `data.seq_len` must be a multiple of `model.gdn_chunk_size` and at
most `model.max_seq_len`; `model.vocab_size` must match the tokenizer and
prepared data; and `train.batch_size` must divide evenly across the device count.

**Gradient checkpointing** (`model.remat`, enabled in the GPU configs): each
decoder layer is recomputed during the backward pass instead of storing its
activations — activation memory stops growing with depth, for ~1/3 extra
forward compute. Gradients are identical to the un-checkpointed forward
(tested); inference (`step`/`generate`) is unaffected.

## Inference API

```python
import flax.nnx as nnx
import jax
from kimi_k3_gdn2 import KimiK3, KimiK3Config

model_cfg = KimiK3Config()
model = KimiK3(model_cfg, rngs=nnx.Rngs(0))
logits, aux = model(input_ids)                      # training: full-sequence, chunkwise-parallel

out = model.generate(prompt_ids, max_new_tokens=128,  # streaming: O(1)/token for GDN-2 layers,
                     temperature=0.8, top_p=0.95,     # O(context) for the few MLA layers
                     eos_id=eos, key=jax.random.PRNGKey(0))
```

`temperature=0` (default) decodes greedily; `eos_id` stops once every batch row
has finished. By default, prompt plus continuation must fit `model_cfg.max_seq_len`;
pass a larger explicit `max_len` only when you intentionally want a larger MLA
cache. Lower-level streaming: `model.init_cache(...)` + `model.step(...)`.

## Tests

```bash
uv run --frozen pytest -m "not multi_device"    # several minutes on CPU

# Exercise the real shard_map training step over two logical CPU devices:
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=2 \
  uv run --frozen pytest -m multi_device tests/test_data_parallel.py -v
```

On PowerShell, set the two-device variables with
`$env:JAX_PLATFORMS='cpu'` and
`$env:XLA_FLAGS='--xla_force_host_platform_device_count=2'` before the test.
GitHub Actions runs formatting, linting, Pyright, the regular CPU suite, and the
two-device smoke test on every push and pull request.

To intentionally refresh dependency versions, run `uv lock --upgrade`, execute
the complete check suite above, and commit the resulting `uv.lock` change with
the compatibility fixes it required. Normal development and CI should keep using
`--frozen` so dependency releases cannot change a run implicitly.

The suite verifies the numerics the docstrings promise: every chunkwise GDN-2
core against a token-by-token scan **and** an independent float64 oracle
(including the documented fp32 overflow limits), the folded GQA recurrence
against the paper's repeat formulation, streaming decode against the training
forward, MoE dispatch against a dense reference, and the Muon/AdamW parameter
split.
