"""Attention Residuals (AttnRes) — attention across DEPTH.

arXiv:2603.15031 (original, §3.1-3.2) / Kimi K3 arXiv:2607.24653 (§2.2).
Transcribed from the original's Fig. 2 pseudocode; see residuals.py.
"""

from attention_residuals.residuals import AttnResReader, RMSNorm, block_attn_res

__all__ = ["AttnResReader", "RMSNorm", "block_attn_res"]
