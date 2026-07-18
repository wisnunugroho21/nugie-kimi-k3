"""
LatentMoE channel mixer (arXiv:2601.18089) — routed experts in a shared low-rank
latent space. This is the MoE design Kimi K3's "Stable LatentMoE" builds on (and
the one adopted by NVIDIA's Nemotron-3 Super/Ultra). In this K3 recreation it is
the REQUIRED channel mixer: every DecoderLayer uses it (kimi_linear_gdn2.py).
The "Stable" additions and Quantile Balancing K3 layers on top are not yet
recreated — they await the K3 technical report.

THE IDEA
--------
A standard MoE runs every routed expert at the full model width d_model, so
expert parameters, expert FLOPs, and (at scale) the all-to-all dispatch traffic
all pay d_model per token. LatentMoE decouples expert computation from the model
width: ONE pair of projections shared by ALL routed experts moves tokens into
and out of a low-rank latent of width d_latent,

    z      = W_down x                        # d_model -> d_latent, shared
    y      = W_up ( Σ_k g_k · f_{e_k}(z) )   # experts + top-k combine IN THE
                                             # LATENT, then one up-projection

with each routed expert f_e a SwiGLU MLP living entirely in the latent
(d_latent -> d_ff -> d_latent). Two consequences (paper §3):

  * Routing stays on the FULL-dimensional x — the router sees the token before
    compression, so no routing signal is lost. The shared (always-on) expert
    also stays full-width; only the routed experts are compressed.
  * At compression α = d_model/d_latent, each expert shrinks by α in both
    params and FLOPs, so the paper's iso-cost recipe ("ℓ-MoE_acc") scales BOTH
    the expert count and top-k by α: total expert parameters, per-token expert
    FLOPs, and communication all match the uncompressed MoE while accuracy
    improves. The paper finds quality preserved for α <= 4 and recommends
    α = 4 — that scaling is how Kimi K3 lands on 16-of-896 experts versus
    K2's 8-of-384. (This repo's demo config mirrors it: the 2-of-8 full-width
    MoE becomes 8-of-32 latent experts at d_latent = d_model/4.)

WHAT IS REUSED
--------------
Routing is INHERITED from GroupedGemmMoE, verbatim: sigmoid affinities, the
aux-loss-free selection bias (`router_bias` + `update_router_bias` in the
training loop), group-limited routing, normalized top-k gate weights, and the
optional Switch-style aux loss. The aux dict keys ("load", "aux_loss",
"group_sizes") are unchanged, so the model/training-loop contract is identical.
The dispatch machinery (sort by expert id -> ragged_dot grouped GEMMs ->
scatter-add combine) is also the same pattern, just run at latent width — the
dispatch/combine memory traffic shrinks by α too.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from multi_latent_attention.moe import _XAVIER, GroupedGemmMoE

F32 = jnp.float32


class LatentMoE(GroupedGemmMoE):
    """Token-dispatched grouped-GEMM MoE whose routed experts live in a shared
    d_latent space (LatentMoE, arXiv:2601.18089), with a full-width shared expert.

    Subclasses GroupedGemmMoE ONLY to inherit its routing (`_route`) and shared
    expert (`_shared`) unchanged; __init__ is written from scratch so the
    full-width expert stacks are never allocated.

    Args (differences from GroupedGemmMoE):
        d_latent: shared latent width ℓ the routed experts operate in
                  (paper recommendation: d_model // 4).
        d_ff:     per-expert hidden width INSIDE the latent. Keeping d_ff and
                  α-scaling n_routed/top_k reproduces the uncompressed MoE's
                  total and active expert cost exactly (see module docstring).
    """

    def __init__(
        self,
        d_model: int,
        d_latent: int,
        d_ff: int,
        n_routed: int = 256,
        n_shared: int = 1,
        top_k: int = 8,
        *,
        n_groups: int = 1,
        topk_groups: int = 1,
        norm_topk: bool = True,
        routed_scale: float = 1.0,
        bias_balancing: bool = True,
        aux_alpha: float = 1e-3,
        compute_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        # NOTE: deliberately NOT calling super().__init__ — it would allocate
        # the full-width expert stacks. Instead set every attribute _route and
        # _shared consume, identically to GroupedGemmMoE.
        assert n_routed % n_groups == 0, "n_routed must be divisible by n_groups"
        assert 1 <= topk_groups <= n_groups, "need 1 <= topk_groups <= n_groups"
        assert top_k <= topk_groups * (n_routed // n_groups), (
            "top_k experts must fit inside the topk_groups selected groups"
        )
        self.d_model = d_model
        self.d_latent = d_latent
        self.d_ff = d_ff
        self.E = n_routed
        self.top_k = top_k
        self.n_groups = n_groups
        self.topk_groups = topk_groups
        self.norm_topk = norm_topk
        self.routed_scale = routed_scale
        self.bias_balancing = bias_balancing
        self.aux_alpha = aux_alpha
        self.compute_dtype = compute_dtype

        # Router on the FULL-dimensional token (paper §3: routing weights are
        # computed from the original x, BEFORE the latent compression) + the
        # aux-loss-free selection bias, exactly as in GroupedGemmMoE.
        self.router = nnx.Linear(
            d_model, n_routed, use_bias=False, kernel_init=_XAVIER, rngs=rngs
        )
        self.router_bias = nnx.Variable(jnp.zeros((n_routed,), F32))

        # Shared latent projections: ONE pair for all routed experts.
        self.w_down = nnx.Linear(
            d_model,
            d_latent,
            use_bias=False,
            kernel_init=_XAVIER,
            dtype=compute_dtype,
            param_dtype=F32,
            rngs=rngs,
        )
        self.w_up = nnx.Linear(
            d_latent,
            d_model,
            use_bias=False,
            kernel_init=_XAVIER,
            dtype=compute_dtype,
            param_dtype=F32,
            rngs=rngs,
        )

        # Stacked routed-expert weights IN THE LATENT: FC1/gate fused into w_in
        # (two grouped GEMMs per forward, as in the parent), fan-in init.
        kin, kout = jax.random.split(rngs.params(), 2)
        self.w_in = nnx.Param(
            jax.random.normal(kin, (n_routed, d_latent, 2 * d_ff), F32)
            * (d_latent**-0.5)
        )
        self.w_out = nnx.Param(
            jax.random.normal(kout, (n_routed, d_ff, d_latent), F32) * (d_ff**-0.5)
        )

        # Shared expert(s): full-width SwiGLU, always-on (paper keeps the shared
        # expert at the original dimension). Same layout as the parent so
        # _shared works unchanged.
        sg, su, sd = jax.random.split(rngs.params(), 3)
        ish = d_ff * n_shared
        self.ws_gate = nnx.Param(
            jax.random.normal(sg, (d_model, ish), F32) * (d_model**-0.5)
        )
        self.ws_up = nnx.Param(
            jax.random.normal(su, (d_model, ish), F32) * (d_model**-0.5)
        )
        self.ws_down = nnx.Param(
            jax.random.normal(sd, (ish, d_model), F32) * (ish**-0.5)
        )

    # ----------------------------------------------------------------------- #
    def __call__(self, x: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Same pipeline as GroupedGemmMoE.__call__ — route / dispatch / grouped
        GEMM / combine — with the expert stage bracketed by the shared latent
        projections: compress ONCE per token, dispatch the latents, and combine
        the top-k expert outputs in the latent before ONE shared up-projection.
        """
        B, L, d = x.shape
        T = B * L
        k = self.top_k
        xf = x.reshape(T, d)
        cdtype = self.compute_dtype

        top_idx, gate, router_logits = self._route(xf)  # inherited, on full-dim x

        # ---- compress to the latent ONCE per token (shared W_down) ----
        z = self.w_down(xf)  # [T, d_latent]

        # ---- dispatch: flatten assignments and sort by expert id ----
        flat_e = top_idx.reshape(T * k).astype(jnp.int32)
        flat_tok = jnp.repeat(jnp.arange(T, dtype=jnp.int32), k)
        flat_w = gate.reshape(T * k).astype(F32)

        order = jnp.argsort(flat_e)  # group same-expert rows
        sort_tok = flat_tok[order]
        sort_w = flat_w[order]
        group_sizes = jnp.bincount(flat_e, length=self.E)  # [E], sums to T*k

        z_sorted = z[sort_tok].astype(cdtype)  # [M, d_latent], M = T*k

        # ---- grouped GEMM in the latent: one matmul per expert ----
        h = jax.lax.ragged_dot(z_sorted, self.w_in.astype(cdtype), group_sizes)
        g_, u_ = jnp.split(h, 2, axis=-1)  # [M, d_ff] each
        a = jax.nn.silu(g_) * u_
        y_sorted = jax.lax.ragged_dot(
            a, self.w_out.astype(cdtype), group_sizes
        )  # [M, d_latent]

        # ---- combine IN THE LATENT, then one shared up-projection ----
        y_sorted = y_sorted.astype(F32) * sort_w[:, None]
        routed_z = (
            jnp.zeros((T, self.d_latent), F32).at[sort_tok].add(y_sorted)
        )  # scatter-add over slots
        routed = self.w_up(routed_z.astype(cdtype)).astype(F32)  # [T, d_model]

        out = routed + self._shared(xf).astype(F32)  # inherited shared expert
        out = out.reshape(B, L, d).astype(cdtype)

        # ---- diagnostics + aux loss: identical contract to the parent ----
        load = group_sizes.astype(F32) / (T * k)
        probs = jax.nn.softmax(router_logits, axis=-1).mean(0)  # [E]
        aux_loss = self.aux_alpha * self.E * jnp.sum(load * probs)
        aux = {"load": load, "aux_loss": aux_loss, "group_sizes": group_sizes}
        return out, aux

    # ----------------------------------------------------------------------- #
    def dense_forward(self, x: jax.Array) -> jax.Array:
        """Reference path computing every latent expert densely (tests only).
        Same weights and routing as __call__, so any mismatch is a dispatch bug."""
        B, L, d = x.shape
        T = B * L
        xf = x.reshape(T, d)
        top_idx, gate, _ = self._route(xf)

        full = (
            jnp.zeros((T, self.E), F32).at[jnp.arange(T)[:, None], top_idx].add(gate)
        )  # [T,E] sparse weights
        z = self.w_down(xf).astype(F32)  # [T, l]
        h = jnp.einsum("tl,elf->tef", z, self.w_in)  # [T, E, 2*d_ff]
        g_, u_ = jnp.split(h, 2, axis=-1)
        a = jax.nn.silu(g_) * u_
        ye = jnp.einsum("tef,efl->tel", a, self.w_out)  # [T, E, l]
        routed_z = jnp.einsum("te,tel->tl", full, ye)
        routed = self.w_up(routed_z)
        out = routed + self._shared(xf)
        return out.reshape(B, L, d)
