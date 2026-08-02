# Kimi K3 in JAX / Flax NNX

A readable, tested implementation of the architecture in
**"Kimi K3: Open Frontier Intelligence"** (Kimi Team, [arXiv:2607.24653](https://arxiv.org/abs/2607.24653)).

The goal is to make §2 of the report executable and explainable: every module
names the equation it implements, and every non-obvious line says *why* the
paper does it that way. The defaults in `config.py` are the real 2.8T model;
`KimiK3Config.tiny()` keeps the same structure at a size that runs on a laptop.

```bash
python demo.py                    # guided tour of every component
python -m pytest tests/ -q        # 30 tests
python -m kimi_k3.config          # parameter-count check against Table 1
```

## The architecture in one paragraph

K3 scales information flow along three axes, one mechanism each:

| Axis | Mechanism | What it replaces |
|---|---|---|
| **Sequence** | Hybrid Attention — 3 Kimi Delta Attention layers per Gated MLA layer | all-global attention |
| **Depth** | Attention Residuals — layers *attend* over previous blocks | the additive residual stream |
| **Width** | Stable LatentMoE — 16 of 896 routed experts, in a half-width latent | a dense FFN |

Plus a native vision pathway at the input (MoonViT-V2) and Per-Head Muon as the
optimizer. Together the report reports ~2.5× the scaling efficiency of Kimi K2.

## Where each part of the paper lives

| Paper | File | Key equations |
|---|---|---|
| §2.1.1 Kimi Delta Attention | [`kda.py`](kimi_k3/kda.py) | Eq. 1 recurrence, Eqs. 3–4 chunkwise form, Eq. 5 lower-bounded decay, Eq. 6 output gate |
| §2.1.2 Gated MLA (NoPE) | [`mla.py`](kimi_k3/mla.py) | Eq. 7 |
| §2.2 Attention Residuals | [`attn_res.py`](kimi_k3/attn_res.py) | Eqs. 8–9 full form, Eq. 10 block form |
| §2.3 Stable LatentMoE | [`moe.py`](kimi_k3/moe.py) | Eq. 11 |
| §2.3.2 SiTU-GLU | [`layers.py`](kimi_k3/layers.py) | Eq. 12, App. B Eqs. 18–19 |
| §2.3.3 Quantile Balancing | [`moe.py`](kimi_k3/moe.py) | Eqs. 13–14, App. C |
| §2.4 Native Vision | [`vision.py`](kimi_k3/vision.py) | — |
| §2.5 Per-Head Muon | [`muon.py`](kimi_k3/muon.py) | — |
| §2, Table 1, MTP, §4.1.4 EAGLE-3 fusion | [`model.py`](kimi_k3/model.py), [`config.py`](kimi_k3/config.py) | Eq. 10 assembly |

## The four ideas, briefly

**Kimi Delta Attention** keeps a fixed-size associative memory `S ∈ R^{d_k×d_v}`
instead of an attention matrix, updated per token by a delta rule with a
*channel-wise* forget gate. K3's change over Kimi Linear is numerical: the
log-decay is bounded below by `g_min = -5` via a scaled sigmoid rather than an
unbounded negative-softplus. The chunkwise training form has to rescale keys by
the *reciprocal* cumulative decay, which previously overflowed; bounding it puts
`1/Γ` under `e^80` for a 16-token tile, inside the BF16 range, so every tile —
diagonal included — becomes a dense matmul.

**Attention Residuals** does to depth what the Transformer did to time. A
standard residual stream folds every layer output in with weight exactly 1, so
by layer 90 the embedding is one summand among ninety. Instead each layer holds
a learned pseudo-query `w_l ∈ R^d` and attends over the outputs of preceding
blocks. Cost: one `d`-vector per layer.

**Stable LatentMoE** runs routed experts in a half-width latent so K3 can afford
896 of them with 16 active, while routing and the shared experts stay full-width.
"Stable" is three fixes for what breaks at that sparsity: an RMSNorm before the
up-projection, SiTU-GLU instead of SwiGLU (bounding both branches of the
product at `β₁β₂ = 100`), and Quantile Balancing.

**Quantile Balancing** replaces the fixed-step load-balancing bias nudge with the
exact bias that produces the target load, read off a quantile of the routing
margins. Appendix C derives it as exact coordinate minimization of the dual of
the balanced-assignment LP — which is why it has no learning rate and settles in
a few steps even with ~10³ experts.

## What is implemented, and what is not

Implemented: everything in §2 (the full architecture), the MTP layer of Table 1,
the EAGLE-3 feature-fusion projection of §4.1.4, Per-Head Muon with the cosine
schedule of §3.3, and both the exact and histogram forms of the QB update.

Not implemented, and out of scope for an architecture reference: the post-training
pipeline (SFT, RL, Multi-Teacher On-Policy Distillation), the MXFP4/MXFP8
quantization-aware training of §4.1.4, the distributed-systems work of §5
(expert-parallel training, sequence partitioning, sandbox infrastructure), and
native-resolution image packing. K2's weight-clipping mechanism, referenced but
not restated by the K3 report, is also absent.

Correctness is checked rather than assumed: the chunkwise KDA form is tested
against the token-by-token recurrence it is derived from, both attention layers'
streaming decode paths are tested against their full forward pass, SiTU-GLU is
tested against its stated bound, and QB is tested to actually reduce load
imbalance.

## A note on unstated hyper-parameters

The report gives total (2.78T) and activated (104.2B) parameter counts but not
the per-head sizes or the MLA latent ranks. Those counts pin them down:
96 heads × 128 dims for both KDA and MLA, `q_lora_rank = 1536`,
`kv_lora_rank = 512`, and shared experts at the routed experts' hidden width
reproduce **2.78T total and 103.5B activated**. Every such value is marked
`[inferred]` in `config.py`, and `python -m kimi_k3.config` prints the arithmetic.

## Requirements

`jax`, `flax` (NNX), `optax`. Developed against JAX 0.10 / Flax 0.12 on CPU;
nothing is device-specific.
