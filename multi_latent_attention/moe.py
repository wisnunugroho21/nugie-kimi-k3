"""
LatentMoE channel mixer (arXiv:2601.18089) — routed experts in a shared low-rank
latent space. This is the MoE design Kimi K3's "Stable LatentMoE" builds on (and
the one adopted by NVIDIA's Nemotron-3 Super/Ultra). In this K3 recreation it is
the REQUIRED channel mixer: every DecoderLayer uses it (kimi_k3_gdn2.py).
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

ROUTING AND DISPATCH
--------------------
Routing uses sigmoid affinities, an aux-loss-free selection bias
(`router_bias` + `update_router_bias` in the training loop), group-limited
routing, normalized top-k gate weights, and an optional DeepSeek-V3-style
sequence-level balancing aux loss. The dispatch machinery sorts assignments by
expert id, applies grouped GEMMs with `ragged_dot`, and scatter-adds the weighted
expert outputs in the latent. The aux dict exposes "load", "aux_loss", and
"group_sizes" for the training loop.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp

F32 = jnp.float32

# App. D.5: Xavier-uniform init with gain 2^{-2.5} (variance_scaling scale =
# gain² = 2^{-5}). Biases stay at zero (the NNX default). The stacked expert
# weights below keep their own explicit fan-in initialization.
_XAVIER = nnx.initializers.variance_scaling(2**-5, "fan_avg", "uniform")


class LatentMoE(nnx.Module):
    """Token-dispatched grouped-GEMM MoE whose routed experts live in a shared
    d_latent space (LatentMoE, arXiv:2601.18089), with a full-width shared expert.

    Args:
        d_model:  model width.
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
        dims = {
            "d_model": d_model,
            "d_latent": d_latent,
            "d_ff": d_ff,
            "n_routed": n_routed,
            "n_shared": n_shared,
            "top_k": top_k,
            "n_groups": n_groups,
            "topk_groups": topk_groups,
        }
        invalid = [name for name, value in dims.items() if value <= 0]
        if invalid:
            raise ValueError(f"MoE dimensions must be positive: {', '.join(invalid)}")
        if d_latent > d_model:
            raise ValueError("d_latent must not exceed d_model")
        if n_routed % n_groups:
            raise ValueError("n_routed must be divisible by n_groups")
        if not 1 <= topk_groups <= n_groups:
            raise ValueError("need 1 <= topk_groups <= n_groups")
        if top_k > topk_groups * (n_routed // n_groups):
            raise ValueError("top_k experts must fit inside the selected expert groups")
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
        # aux-loss-free selection bias.
        self.router = nnx.Linear(d_model, n_routed, use_bias=False, kernel_init=_XAVIER, rngs=rngs)
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
        # (two grouped GEMMs per forward), fan-in init.
        kin, kout = jax.random.split(rngs.params(), 2)
        self.w_in = nnx.Param(
            jax.random.normal(kin, (n_routed, d_latent, 2 * d_ff), F32) * (d_latent**-0.5)
        )
        self.w_out = nnx.Param(
            jax.random.normal(kout, (n_routed, d_ff, d_latent), F32) * (d_ff**-0.5)
        )

        # Shared expert(s): full-width SwiGLU, always-on (paper keeps the shared
        # expert at the original dimension).
        sg, su, sd = jax.random.split(rngs.params(), 3)
        ish = d_ff * n_shared
        self.ws_gate = nnx.Param(jax.random.normal(sg, (d_model, ish), F32) * (d_model**-0.5))
        self.ws_up = nnx.Param(jax.random.normal(su, (d_model, ish), F32) * (d_model**-0.5))
        self.ws_down = nnx.Param(jax.random.normal(sd, (ish, d_model), F32) * (ish**-0.5))

    # ----------------------------------------------------------------------- #
    def _route(self, x_flat: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Route tokens and return top expert indices, gate weights, and logits.

        The aux-loss-free bias affects expert selection only. Gate weights come
        from the true sigmoid affinities so the router retains exact gradients.
        Group-limited routing restricts candidates to each token's best expert
        groups before the final top-k selection.
        """
        logits = self.router(x_flat).astype(F32)
        scores = jax.nn.sigmoid(logits)

        sel = scores + self.router_bias if self.bias_balancing else scores
        sel = jax.lax.stop_gradient(sel) if self.bias_balancing else sel

        if self.n_groups > 1:
            T = sel.shape[0]
            gsize = self.E // self.n_groups
            sel_g = sel.reshape(T, self.n_groups, gsize)
            top2, _ = jax.lax.top_k(sel_g, min(2, gsize))
            group_score = top2.sum(-1)
            _, gidx = jax.lax.top_k(group_score, self.topk_groups)
            keep = jnp.zeros((T, self.n_groups), bool).at[jnp.arange(T)[:, None], gidx].set(True)
            sel = jnp.where(jnp.repeat(keep, gsize, axis=-1), sel, -jnp.inf)

        _, top_idx = jax.lax.top_k(sel, self.top_k)
        gate = jnp.take_along_axis(scores, top_idx, axis=-1)
        if self.norm_topk:
            gate = gate / (gate.sum(-1, keepdims=True) + 1e-9)
        gate = gate * self.routed_scale
        return top_idx, gate, logits

    @staticmethod
    def _norm_sigmoid_probs(router_logits: jax.Array) -> jax.Array:
        """Return the mean normalized sigmoid affinity for each expert."""
        scores = jax.nn.sigmoid(router_logits)
        return (scores / (scores.sum(-1, keepdims=True) + 1e-9)).mean(0)

    def _shared(self, x_flat: jax.Array) -> jax.Array:
        """Apply the always-on, full-width shared SwiGLU expert."""
        cd = self.compute_dtype
        xf = x_flat.astype(cd)
        a = jax.nn.silu(xf @ self.ws_gate.astype(cd)) * (xf @ self.ws_up.astype(cd))
        return a @ self.ws_down.astype(cd)

    # ----------------------------------------------------------------------- #
    def __call__(self, x: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Route, dispatch, run latent experts, and combine their outputs.

        The shared projections compress each token once before dispatch and
        expand the combined latent output once after the expert computation.
        """
        B, L, d = x.shape
        T = B * L
        k = self.top_k
        xf = x.reshape(T, d)
        cdtype = self.compute_dtype

        top_idx, gate, router_logits = self._route(xf)  # routing stays full-width

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
        y_sorted = jax.lax.ragged_dot(a, self.w_out.astype(cdtype), group_sizes)  # [M, d_latent]

        # ---- combine IN THE LATENT, then one shared up-projection ----
        y_sorted = y_sorted.astype(F32) * sort_w[:, None]
        routed_z = (
            jnp.zeros((T, self.d_latent), F32).at[sort_tok].add(y_sorted)
        )  # scatter-add over slots
        routed = self.w_up(routed_z.astype(cdtype)).astype(F32)  # [T, d_model]

        out = routed + self._shared(xf).astype(F32)
        out = out.reshape(B, L, d).astype(cdtype)

        # ---- diagnostics + aux loss ----
        load = group_sizes.astype(F32) / (T * k)
        probs = self._norm_sigmoid_probs(router_logits)  # [E]
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


# --------------------------------------------------------------------------- #
def update_router_bias(bias: jax.Array, group_sizes: jax.Array, lr: float = 1e-3) -> jax.Array:
    """Nudge selection bias toward uniformly loaded experts after a step."""
    total = jnp.sum(group_sizes).astype(F32)
    load = group_sizes.astype(F32) / jnp.maximum(total, 1.0)
    target = 1.0 / bias.shape[0]
    return bias + lr * jnp.sign(target - load)
