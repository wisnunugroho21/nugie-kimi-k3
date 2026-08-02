"""Kimi K3 configuration — the numbers behind Table 1 of the technical report.

Paper: "Kimi K3: Open Frontier Intelligence" (Kimi Team, arXiv:2607.24653).

Everything in `KimiK3Config` that the report states explicitly is marked
"[Table 1]" or with its section number. A handful of dimensions are NOT stated
in the report (per-head sizes, the MLA latent ranks); those are marked
"[inferred]" and the reasoning is spelled out in `_PARAM_COUNT_CHECK` at the
bottom of this file — the inferred values are the ones that reproduce the
report's own headline parameter counts to within 1%.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class KimiK3Config:
    """Architecture hyper-parameters for the Kimi K3 backbone.

    The defaults are the real 2.8T model. Nothing that size will run on a
    laptop, so use `KimiK3Config.tiny()` for anything you actually execute; the
    tiny config keeps every structural ratio of the real model (3:1 KDA:MLA,
    one trailing MLA layer, one leading dense layer, 8 AttnRes blocks) and only
    shrinks the widths.
    """

    # ---------------------------------------------------------------- backbone
    vocab_size: int = 160_000  # [Table 1] 160K
    hidden_size: int = 7_168  # [Table 1] d_model, unchanged from K2
    num_layers: int = 93  # [Table 1] 93 = 69 KDA + 24 MLA

    # -------------------------------------------------- §2.1 Hybrid Attention
    # "Each block contains 3 KDA layers followed by 1 Gated MLA layer, giving a
    # 3:1 mixing ratio... An additional Gated MLA layer is placed at the end of
    # the backbone, ensuring that the final layer always performs global
    # attention."  With 93 layers that is 23 * (3 KDA + 1 MLA) + 1 MLA
    # = 69 KDA + 24 MLA, exactly the split reported in Table 1.
    kda_per_mla: int = 3

    # [Table 1] "Attention Heads 96". The report does not break this down per
    # mechanism; we use 96 heads for BOTH KDA and MLA, with head_dim 128
    # inherited from K2. See `_PARAM_COUNT_CHECK` — this choice is what makes
    # the activated-parameter count come out at ~104B.
    num_heads: int = 96  # [Table 1]
    head_dim: int = 128  # [inferred] d_k = d_v per head

    # KDA specifics (§2.1.1)
    kda_conv_size: int = 4  # ShortConv window; standard depthwise causal conv
    kda_alpha_rank: int = 128  # rank of the low-rank decay projection W_a^up W_a^down
    kda_g_min: float = -5.0  # [§2.1.1, Eq. 5] the lower bound on log-decay

    # Gated MLA specifics (§2.1.2). Latent ranks are not in the report; these
    # are K2's / DeepSeek-V3's values, which the parameter check corroborates.
    q_lora_rank: int = 1_536  # [inferred] query down-projection rank
    kv_lora_rank: int = 512  # [inferred] the cached KV latent c_t = W_c x_t

    # ------------------------------------------------ §2.2 Attention Residuals
    # "for Kimi K3, we partition its layers into 8 blocks with 12-layer size,
    # giving a partial final block and 9 total blocks when counting the
    # embedding layer."  8 * 12 = 96 >= 93, so the last block holds 9 layers.
    attn_res_block_size: int = 12

    # ------------------------------------------------- §2.3 Stable LatentMoE
    num_routed_experts: int = 896  # [Table 1]
    num_experts_per_token: int = 16  # [Table 1] top-k; sparsity 896/16 = 56
    num_shared_experts: int = 2  # [Table 1]
    moe_latent_size: int = 3_584  # [Table 1] "Latent MoE Dimension 3584 (0.5x)"
    moe_expert_hidden: int = 3_072  # [Table 1] "MoE Hidden Dimension per Expert"
    num_dense_layers: int = 1  # [Table 1] layer 0 gets a plain full-width FFN
    # Width of that one dense layer's FFN. Not stated in the report; this is the
    # K2 / DeepSeek-V3 value for d_model = 7168.
    dense_ffn_hidden: int = 18_432  # [inferred]

    # SiTU-GLU soft caps (§2.3.2, Eq. 12). "we set the soft-cap hyperparameters
    # to beta1 = 4 for the gate branch and beta2 = 25 for the up branch",
    # bounding every output coordinate by beta1*beta2 = 100 (Appendix B, Eq. 19).
    situ_beta_gate: float = 4.0
    situ_beta_up: float = 25.0

    # ------------------------------------------------------------------- misc
    num_mtp_layers: int = 1  # [Table 1] one multi-token-prediction layer
    rms_norm_eps: float = 1e-6
    # KDA chunk length for the chunkwise-parallel form (§2.1.1, Eqs. 3-4).
    # 16 is the report's secondary tile size: with g_min = -5 the cumulative
    # log-decay over 16 tokens lies in (-80, 0), so the reciprocal rescaling
    # factor stays inside the BF16 dynamic range.
    kda_chunk_size: int = 16

    # ------------------------------------------------------ §2.4 Native Vision
    vision_layers: int = 27  # [Table 1] "#ViT Layers 27"
    vision_hidden: int = 1_152  # [inferred] SigLIP-So400M width; gives ~411M
    vision_mlp_hidden: int = 4_304  # [inferred] same lineage
    vision_heads: int = 12  # [Table 1] "#Attention Heads of ViT 12"
    vision_patch_size: int = 14  # [Table 1] "Patch Size of ViT 14"
    vision_pixel_shuffle: int = 2  # §2.4: 2x2 downsample before the projector
    vision_temporal_pool: int = 2  # §2.4: temporal pooling across frames

    # ------------------------------------------------------------- properties
    @property
    def num_attn_res_blocks(self) -> int:
        """N in §2.2 — how many 12-layer blocks the depth splits into (ceil)."""
        return -(-self.num_layers // self.attn_res_block_size)

    def is_mla_layer(self, layer_idx: int) -> bool:
        """§2.1: every (kda_per_mla+1)-th layer is global, and so is the last one."""
        last = layer_idx == self.num_layers - 1
        periodic = layer_idx % (self.kda_per_mla + 1) == self.kda_per_mla
        return last or periodic

    def is_dense_layer(self, layer_idx: int) -> bool:
        """[Table 1] the first `num_dense_layers` layers skip the MoE."""
        return layer_idx < self.num_dense_layers

    # ------------------------------------------------------------- factories
    @classmethod
    def tiny(cls, **overrides) -> "KimiK3Config":
        """A ~10M-parameter model with the real model's structure. For tests.

        9 layers => 6 KDA + 3 MLA (layers 3, 7 periodic + layer 8 trailing),
        3 AttnRes blocks of 3 layers, 1 leading dense layer.
        """
        base = dict(
            vocab_size=512,
            hidden_size=128,
            num_layers=9,
            num_heads=4,
            head_dim=32,
            q_lora_rank=64,
            kv_lora_rank=32,
            kda_alpha_rank=16,
            attn_res_block_size=3,
            num_routed_experts=16,
            num_experts_per_token=4,
            num_shared_experts=2,
            moe_latent_size=64,
            moe_expert_hidden=64,
            dense_ffn_hidden=256,
            vision_layers=2,
            vision_hidden=64,
            vision_mlp_hidden=128,
            vision_heads=4,
            vision_patch_size=8,
        )
        base.update(overrides)
        return cls(**base)


# --------------------------------------------------------------------------
# Why the "[inferred]" numbers above are the right ones.
#
# The report gives total (2.78T) and activated (104.2B) parameter counts but
# not the per-head sizes or MLA ranks. Those counts are enough to pin them
# down. Run this file (`python -m kimi_k3.config`) to see the arithmetic.
# --------------------------------------------------------------------------
_PARAM_COUNT_CHECK = """
With num_heads=96, head_dim=128, q_lora_rank=1536, kv_lora_rank=512 and
shared experts at the routed experts' hidden width (3072):

  KDA layer   3*d*(H*Dh)              qkv projections
            + (H*Dh)*d                output projection
            + d*(H*Dh)                full-rank output gate (Eq. 6)      ~440M
  MLA layer   d*q_lora + q_lora*H*Dh  query down/up
            + d*kv_lora               the cached latent c_t = W_c x_t
            + kv_lora*H*2*Dh          key/value up-projections
            + (H*Dh)*d + d*(H*Dh)     output projection + gate (Eq. 7)   ~222M
  MoE layer   k*3*l*d_ff              16 active routed experts (latent)
            + 2*3*d*d_ff              2 full-width shared experts
            + 2*d*l                   W_down / W_up                      ~712M

  activated = 69*KDA + 24*MLA + 92*MoE_active + 2*V*d  = 103.5B  (paper 104.2B)
  total     = 69*KDA + 24*MLA + 92*MoE_total  + 2*V*d  =   2.78T  (paper 2.78T)

The total is dominated by the routed experts: 896 experts * 3 matrices *
3584 * 3072 * 92 layers = 2.72T on its own, which is what fixes the latent
and expert-hidden widths as stated in Table 1.
"""


def _report() -> None:  # pragma: no cover - documentation helper
    c = KimiK3Config()
    d, hd, lat, dff = c.hidden_size, c.num_heads * c.head_dim, c.moe_latent_size, c.moe_expert_hidden
    n_mla = sum(c.is_mla_layer(i) for i in range(c.num_layers))
    n_kda = c.num_layers - n_mla
    n_moe = c.num_layers - c.num_dense_layers

    kda = 3 * d * hd + hd * d + d * hd
    mla = d * c.q_lora_rank + c.q_lora_rank * hd + d * c.kv_lora_rank
    mla += c.kv_lora_rank * c.num_heads * 2 * c.head_dim + hd * d + d * hd
    moe_act = c.num_experts_per_token * 3 * lat * dff + c.num_shared_experts * 3 * d * dff + 2 * d * lat
    moe_tot = c.num_routed_experts * 3 * lat * dff + c.num_shared_experts * 3 * d * dff + 2 * d * lat
    emb = 2 * c.vocab_size * d

    print(_PARAM_COUNT_CHECK)
    print(f"  layers: {n_kda} KDA + {n_mla} MLA = {c.num_layers}   (paper: 69 + 24 = 93)")
    print(f"  AttnRes blocks: {c.num_attn_res_blocks}              (paper: 8, +1 for the embedding)")
    print(f"  activated: {(n_kda * kda + n_mla * mla + n_moe * moe_act + emb) / 1e9:.1f}B  (paper: 104.2B)")
    print(f"  total:     {(n_kda * kda + n_mla * mla + n_moe * moe_tot + emb) / 1e12:.2f}T  (paper: 2.78T)")


if __name__ == "__main__":  # pragma: no cover
    _report()
