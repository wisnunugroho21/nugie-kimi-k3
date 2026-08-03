"""A JAX / Flax NNX implementation of Kimi K3.

Paper: "Kimi K3: Open Frontier Intelligence", Kimi Team, arXiv:2607.24653.

Section-to-file map:

    §2.1.1  Kimi Delta Attention .................... kda.py
    §2.1.2  Gated MLA (NoPE) ....................... mla.py
    §2.2    Attention Residuals .................... attn_res.py
    §2.3    Stable LatentMoE ....................... moe.py
              §2.3.1 Normalized LatentMoE
              §2.3.2 SiTU-GLU ...................... layers.py
              §2.3.3 Quantile Balancing
    §2.4    Native Vision (MoonViT-V2) ............. vision.py
    §2.5    Per-Head Muon .......................... muon.py
    Table 1 Hyper-parameters ....................... config.py
    §2      Assembly, MTP layer .................... model.py
"""

from .attn_res import AttnResStream, DepthAttention
from .config import KimiK3Config
from .kda import KimiDeltaAttention, kda_chunkwise, kda_recurrent
from .layers import ShortConv, SiTUGLU, softcap
from .mla import GatedMLA
from .model import (
    DecoderLayer,
    Eagle3Fusion,
    KimiK3,
    MTPLayer,
    apply_quantile_balancing,
    causal_lm_loss,
    mtp_loss,
)
from .moe import (
    RoutedExperts,
    StableLatentMoE,
    quantile_balancing_update,
    quantile_balancing_update_histogram,
)
from .muon import cosine_schedule_with_warmup, kimi_k3_optimizer, newton_schulz, per_head_muon
from .vision import MoonViTV2, VisionTower

__all__ = [
    "AttnResStream",
    "DecoderLayer",
    "DepthAttention",
    "Eagle3Fusion",
    "GatedMLA",
    "KimiDeltaAttention",
    "KimiK3",
    "KimiK3Config",
    "MTPLayer",
    "MoonViTV2",
    "RoutedExperts",
    "ShortConv",
    "SiTUGLU",
    "StableLatentMoE",
    "VisionTower",
    "apply_quantile_balancing",
    "causal_lm_loss",
    "cosine_schedule_with_warmup",
    "kda_chunkwise",
    "kda_recurrent",
    "kimi_k3_optimizer",
    "mtp_loss",
    "newton_schulz",
    "per_head_muon",
    "quantile_balancing_update",
    "quantile_balancing_update_histogram",
    "softcap",
]
