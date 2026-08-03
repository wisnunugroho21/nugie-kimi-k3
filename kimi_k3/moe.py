"""Stable LatentMoE — §2.3 of the Kimi K3 report.

K3 activates 16 of 896 routed experts per token: a sparsity of 56, against K2's
384/8 = 48 experts at a sparsity of 48. Two questions follow, and §2.3 is the
answer to both.

WHY THE EXPERTS CAN BE THAT NUMEROUS: LatentMoE
------------------------------------------------
In a conventional MoE every selected expert receives the full d-dimensional
token, so expert parameters, expert FLOPs and all-to-all dispatch traffic all
scale with `d * top_k`. Doubling top_k doubles all three. LatentMoE decouples
the model width from the *routed-expert* width: one shared pair of projections
moves tokens in and out of a compact latent of width `l`, and the routed experts
live entirely inside it (Eq. 11):

    u = sum_{i in TopK(x)} p_i * E_i^routed(W_down x)        experts see R^l
    y = sum_j E_j^shared(x) + W_up RMSNorm(u)                shared see R^d

with l = 3584 = 0.5 * d. Every routed expert is half as wide, so K3 buys twice
the experts and twice the top-k at the same cost. Two details matter:

  * ROUTING STAYS ON THE FULL-WIDTH x (Eq. 13). The router sees the token before
    compression, so no routing signal is lost to the bottleneck.
  * The SHARED experts stay full-width. They carry the transformations every
    token needs; only the specialists are compressed.

WHY IT NEEDS STABILISING: the three "Stable" components
--------------------------------------------------------
At 2.8T scale this design has two failure modes, and §2.3 fixes them with three
pieces, each implemented below:

  1. The routed path chains W_down -> gated FFN -> W_up into nearly four
     consecutive matmuls, an ill-conditioned stack that produced exploding
     activations. Fix: an RMSNorm between expert aggregation and W_up
     (§2.3.1, and it improves validation loss on top of stabilising training).
  2. Same explosion risk inside the expert FFN itself. Fix: SiTU-GLU replaces
     SwiGLU, bounding both branches of the product (§2.3.2, in layers.py).
  3. Balancing ~10^3 experts is past the regime where fixed-step auxiliary-loss-
     free bias updates behave. Fix: Quantile Balancing (§2.3.3), which jumps
     straight to the bias that produces the target load instead of nudging
     toward it.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx

from .config import KimiK3Config
from .layers import F32, SiTUGLU, softcap


def _init_expert(key, shape, fan_in):
    """Truncated-normal init scaled by 1/sqrt(fan_in), per expert matrix."""
    return jax.random.truncated_normal(key, -2.0, 2.0, shape, dtype=F32) * (fan_in**-0.5)


class RouterBias(nnx.Variable):
    """The load-balancing bias `b` of Eq. 13.

    A distinct Variable type, not an `nnx.Param`, because it is NOT trained by
    gradient descent: it is overwritten each step by the Quantile Balancing rule.
    Optimisers in this repo filter on `nnx.Param`, so this is left alone.
    """


class RouterStats(NamedTuple):
    """Everything the Quantile Balancing update needs, collected per layer."""

    scores: jax.Array  # [N, E]  raw sigmoid router scores s (no bias)
    cutoff: jax.Array  # [N]     alpha_i: the (k+1)-th largest biased score
    load: jax.Array  # [E]     tokens dispatched to each expert this batch


# ===========================================================================
# §2.3.3 Quantile Balancing
# ===========================================================================


def quantile_balancing_update(stats: RouterStats, top_k: int) -> jax.Array:
    """Eq. 14 — the new expert bias, computed from one forward pass.

    BACKGROUND. Auxiliary-loss-free balancing (K2's method) keeps a per-expert
    bias `b_j` that is added to the router score for Top-k *selection only*, and
    nudges it by a fixed step: `b_j += gamma * sign(target_load - load_j)`.
    `gamma` then trades slow adaptation against oscillation, and with 896 experts
    per layer there is no good setting.

    QB replaces the nudge with the exact answer. Route with Top-(k+1) instead of
    Top-k: the first k entries are the routes actually taken, and the (k+1)-th
    biased score is the *cutoff* `alpha_i` — the bar an expert must clear to
    enter token i's Top-k. With the cutoffs held fixed, the load of expert j
    under a candidate bias is

        #{ i : s_ij + b_j > alpha_i }

    which is monotonically decreasing in `-b_j`. Setting it equal to the target
    load q = m*k/n makes `-b_j` the (q+1)-th largest margin `s_ij - alpha_i`,
    i.e. the (1 - k/n)-quantile of that expert's margins:

        b_hat_j = -quantile_{1-k/n}( s_:,j - alpha )
        b       = b_hat - mean(b_hat)

    The mean subtraction removes a common offset, which cannot change any Top-k
    decision. Appendix C derives this as exact coordinate minimisation of the
    dual of the maximum-score balanced-assignment LP — which is why there is no
    learning rate to tune and why it equilibrates in a few steps.

    CAUSALITY: the returned bias must be applied on the NEXT step. A batch is
    never routed with a bias derived from itself.
    """
    n_experts = stats.scores.shape[-1]
    margins = stats.scores - stats.cutoff[:, None]  # [N, E]
    # The quantile is taken down the token axis, independently per expert.
    b_hat = -jnp.quantile(margins.astype(F32), 1.0 - top_k / n_experts, axis=0)
    return b_hat - b_hat.mean()


def quantile_balancing_update_histogram(
    stats: RouterStats,
    top_k: int,
    *,
    num_bins: int = 512,
    value_range: tuple[jax.Array, jax.Array] | None = None,
) -> jax.Array:
    """The estimator K3 actually uses in training (§2.3.3, "Histogram estimation").

    The quantile in Eq. 14 is over the whole global batch: millions of margins,
    spread across data-parallel ranks and gradient-accumulation micro-steps.
    Gathering them to sort exactly is not viable. Instead each rank bins its own
    margins per expert; a single all-reduce sums the bin counts; the quantile is
    read off the pooled histogram.

    The trick is that counts are ADDITIVE, so the pooled histogram represents the
    true global batch no matter how tokens were sharded — the estimate is exact
    up to one bin width, at a communication cost of a few hundred numbers per
    expert instead of millions.

    THE BIN GRID HAS TO FOLLOW THE DATA.  Margins are `s_ij - alpha_i`, and the
    cutoff `alpha_i` is taken from the *biased* score, so the margins sit around
    `-b_j` and drift with the bias as training proceeds. A fixed grid is
    therefore wrong: once the biases spread out, every margin lands in the edge
    bins and the estimate collapses to a constant. The range is derived per
    expert from the margins themselves, which keeps the "exact up to one bin
    width" guarantee unconditional.

    DISTRIBUTED USE.  Counts are only additive if every rank bins on the SAME
    grid, so the range must be agreed before binning — a min/max all-reduce of
    two floats per expert, negligible beside the `num_bins` per expert that the
    count all-reduce already costs:

        lo = jax.lax.pmin(margins.min(axis=0), "dp")
        hi = jax.lax.pmax(margins.max(axis=0), "dp")
        bias = quantile_balancing_update_histogram(stats, k, value_range=(lo, hi))
        # ... with `counts = jax.lax.psum(counts, "dp")` at the marker below.

    Pass `value_range` to supply that agreed grid (or any fixed grid you can
    bound analytically); omit it on a single device to derive it locally.
    """
    n_experts = stats.scores.shape[-1]
    margins = (stats.scores - stats.cutoff[:, None]).astype(F32)  # [N, E]

    if value_range is None:
        lo, hi = margins.min(axis=0), margins.max(axis=0)  # [E] each
    else:
        lo, hi = (jnp.broadcast_to(jnp.asarray(v, F32), (n_experts,)) for v in value_range)
    # A degenerate span (every margin identical) would divide by zero; widening
    # it puts all the mass in bin 0, which reads back as `lo` — the right answer.
    span = jnp.maximum(hi - lo, 1e-6)

    # Arithmetic binning rather than searchsorted, because the edges are now
    # per expert: bin b of expert j covers [lo_j + b*span_j/B, lo_j + (b+1)*...).
    idx = jnp.clip(
        jnp.floor((margins - lo) / span * num_bins).astype(jnp.int32), 0, num_bins - 1
    )
    # counts[b, j] = how many of expert j's margins fall in bin b
    counts = jnp.zeros((num_bins, n_experts), dtype=jnp.int32).at[idx, jnp.arange(n_experts)].add(1)
    # ---- a single all-reduce would go here: counts = jax.lax.psum(counts, "dp")

    total = counts.sum(axis=0)  # = N, per expert
    target_rank = jnp.floor((1.0 - top_k / n_experts) * total).astype(jnp.int32)
    cdf = jnp.cumsum(counts, axis=0)  # [num_bins, E]
    bin_idx = (cdf < target_rank[None, :]).sum(axis=0)  # first bin reaching the rank
    # Interpolate inside the located bin instead of taking its lower edge: the
    # rank is known exactly, so this recovers most of the within-bin resolution.
    below = jnp.take_along_axis(cdf, jnp.clip(bin_idx - 1, 0, num_bins - 1)[None], axis=0)[0]
    below = jnp.where(bin_idx == 0, 0, below)
    in_bin = jnp.take_along_axis(counts, bin_idx[None], axis=0)[0]
    frac = jnp.where(in_bin > 0, (target_rank - below) / jnp.maximum(in_bin, 1), 0.0)
    b_hat = -(lo + (bin_idx + frac) * span / num_bins)
    return b_hat - b_hat.mean()


# ===========================================================================
# Experts
# ===========================================================================


class SiTUFFN(nnx.Module):
    """A dense SiTU-GLU feed-forward network: the shared experts, and the FFN of
    K3's one dense layer.  down(SiTU-GLU(x))."""

    def __init__(self, d_in: int, d_hidden: int, cfg: KimiK3Config, *, rngs: nnx.Rngs):
        self.act = SiTUGLU(
            d_in, d_hidden, beta_gate=cfg.situ_beta_gate, beta_up=cfg.situ_beta_up, rngs=rngs
        )
        self.down = nnx.Linear(d_hidden, d_in, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.down(self.act(x))


class RoutedExperts(nnx.Module):
    """The 896 routed experts, stored as stacked weights and run as grouped GEMMs.

    Each expert is a SiTU-GLU FFN `R^l -> R^d_ff -> R^l` living entirely in the
    MoE latent. Weights are held as one [E, ...] tensor per matrix so the whole
    bank is three `ragged_dot`s rather than 896 separate matmuls.

    DISPATCH.  Each of N tokens picks k experts, giving N*k (token, expert)
    assignments. We sort those assignments by expert id, which lays out all rows
    destined for expert 0 contiguously, then expert 1, and so on. `ragged_dot`
    then walks the sorted rows and applies each expert's weight matrix to its own
    contiguous group — one kernel for the entire bank, no padding, no masking.
    The inverse permutation puts results back in token order. (At K3's scale the
    sort is also what the expert-parallel all-to-all is built on: the sorted
    layout is the send buffer.)
    """

    def __init__(self, cfg: KimiK3Config, *, rngs: nnx.Rngs):
        E, l, f = cfg.num_routed_experts, cfg.moe_latent_size, cfg.moe_expert_hidden
        self.E, self.beta_gate, self.beta_up = E, cfg.situ_beta_gate, cfg.situ_beta_up
        k1, k2, k3 = jax.random.split(rngs.params(), 3)
        self.w_gate = nnx.Param(_init_expert(k1, (E, l, f), l))
        self.w_up = nnx.Param(_init_expert(k2, (E, l, f), l))
        self.w_down = nnx.Param(_init_expert(k3, (E, f, l), f))

    def __call__(self, z: jax.Array, expert_idx: jax.Array, weights: jax.Array) -> jax.Array:
        """z: [N, l] latent tokens. expert_idx/weights: [N, k]. Returns u: [N, l]."""
        N, k = expert_idx.shape
        flat_idx = expert_idx.reshape(-1)  # [N*k]

        # Sort assignments by expert id -> contiguous groups for ragged_dot.
        order = jnp.argsort(flat_idx)
        group_sizes = jnp.bincount(flat_idx, length=self.E).astype(jnp.int32)

        rows = jnp.repeat(z, k, axis=0)[order]  # [N*k, l], grouped by expert

        # Three grouped GEMMs = every expert's SiTU-GLU, in three kernels.
        g = jax.lax.ragged_dot(rows, self.w_gate[...], group_sizes)
        up = jax.lax.ragged_dot(rows, self.w_up[...], group_sizes)
        h = (softcap(g, self.beta_gate) * jax.nn.sigmoid(g)) * softcap(up, self.beta_up)
        out = jax.lax.ragged_dot(h, self.w_down[...], group_sizes)  # [N*k, l]

        # Un-sort, then combine the k experts of each token with the router
        # weights p_i (Eq. 11's `sum_i p_i E_i(...)`).
        out = jnp.zeros_like(out).at[order].set(out)
        return jnp.einsum("nk,nkl->nl", weights.astype(F32), out.reshape(N, k, -1))


# ===========================================================================
# The layer
# ===========================================================================


class StableLatentMoE(nnx.Module):
    """Eq. 11 plus the three stabilisers. The channel mixer of every K3 layer."""

    def __init__(self, cfg: KimiK3Config, *, rngs: nnx.Rngs):
        self.cfg = cfg
        d, l = cfg.hidden_size, cfg.moe_latent_size
        self.top_k = cfg.num_experts_per_token

        # --- router: sigmoid affinities over the FULL-width token (Eq. 13).
        # Sigmoid rather than softmax, following K2/DeepSeek-V3: expert scores
        # are independent, so adding one expert does not rescale the others.
        self.router = nnx.Linear(d, cfg.num_routed_experts, use_bias=False, rngs=rngs)
        self.router_bias = RouterBias(jnp.zeros((cfg.num_routed_experts,), dtype=F32))

        # --- the latent path.
        self.down_proj = nnx.Linear(d, l, use_bias=False, rngs=rngs)  # W_down
        self.experts = RoutedExperts(cfg, rngs=rngs)
        # §2.3.1: RMSNorm between aggregation and up-projection. `u`'s scale
        # swings with which experts fired and how the weights fell; this pins it
        # down before the routed branch meets the full-width shared branch.
        self.u_norm = nnx.RMSNorm(l, epsilon=cfg.rms_norm_eps, rngs=rngs)
        self.up_proj = nnx.Linear(l, d, use_bias=False, rngs=rngs)  # W_up

        # --- shared experts: always on, full width. (Summing n independent
        # SiTU-GLU FFNs is algebraically one FFN with n*d_ff hidden units; kept
        # separate here because that is how Eq. 11 reads.)
        self.shared = nnx.List(
            [SiTUFFN(d, cfg.moe_expert_hidden, cfg, rngs=rngs) for _ in range(cfg.num_shared_experts)]
        )

    def __call__(self, x: jax.Array) -> tuple[jax.Array, RouterStats]:
        """x: [B, T, d] -> (y: [B, T, d], stats for the QB update)."""
        B, T, d = x.shape
        flat = x.reshape(B * T, d)

        # ---- Eq. 13: route on the full-width token -----------------------
        scores = jax.nn.sigmoid(self.router(flat).astype(F32))  # s in (0,1)^E
        biased = scores + self.router_bias[...]

        # Top-(k+1): the first k are the routes taken; the (k+1)-th biased score
        # is the cutoff alpha_i that Quantile Balancing needs. Taking it from the
        # same selection avoids a second pass over the scores.
        _, top_idx = jax.lax.top_k(biased, self.top_k + 1)
        expert_idx = top_idx[:, : self.top_k]
        cutoff = jnp.take_along_axis(biased, top_idx[:, self.top_k :], axis=-1)[:, 0]

        # Gate weights come from the UNBIASED scores: `b` regulates dispatch
        # only, and never enters the mixture weights or the router's gradient.
        sel = jnp.take_along_axis(scores, expert_idx, axis=-1)  # [N, k]
        p = sel / jnp.clip(sel.sum(axis=-1, keepdims=True), 1e-9)

        # ---- Eq. 11: routed branch, in the latent -------------------------
        u = self.experts(self.down_proj(flat), expert_idx, p)
        routed = self.up_proj(self.u_norm(u))  # W_up RMSNorm(u)

        # ---- Eq. 11: shared branch, at full width -------------------------
        y = routed
        for expert in self.shared:
            y = y + expert(flat)

        load = jnp.bincount(expert_idx.reshape(-1), length=self.cfg.num_routed_experts)
        return y.reshape(B, T, d), RouterStats(scores, cutoff, load)
