"""
Stable LatentMoE — the Kimi K3 channel mixer (§2.3), in JAX / Flax NNX.

WHAT PROBLEM IT SOLVES
----------------------
K3 wants a much larger expert pool AND more experts per token: 896 routed experts
with 16 active, a sparsity of 56 (K2 had 384/8). In a conventional MoE every
selected expert receives the FULL d-dimensional token, so both the all-to-all
communication and the expert-weight traffic grow with how many experts a token
activates — 16 full-width experts per token is simply too expensive.

LatentMoE separates the model width from the ROUTED-expert width. Shared experts
keep a full-width path for the common transformations; the routed experts work in
a compact latent of width ℓ (K3: ℓ = 0.5·d):

    u = Σ_{i ∈ T_k(x)}  p_i · E_i^routed( W↓ x )                          Eq. 11
    y = Σ_{j=1..N_s}    E_j^shared( x )   +   W↑ RMSNorm(u)

with W↓: R^d -> R^ℓ, each routed expert E_i^routed: R^ℓ -> R^ℓ, W↑: R^ℓ -> R^d,
and N_s = 2 full-width shared experts in every layer. Only the ℓ-wide `z = W↓x`
is dispatched, so doubling the active-expert count no longer doubles the traffic.

WHY "STABLE" — the three additions (§2.3)
-----------------------------------------
Extreme sparsity amplifies two failure modes, which K3 fixes with three pieces:

 1. RMSNorm before W↑ (§2.3.1). The routed path chains W↓ -> a gated multi-branch
    FFN -> W↑, i.e. ~four consecutive matmuls with nothing normalizing in
    between; at 2.8T scale that ill-conditioned chain produced exploding
    activations. Worse, u's SCALE varies with which experts were selected and
    with their routing weights, so the routed branch's contribution to the sum in
    Eq. 11 is not comparable across tokens. Normalizing u before the
    up-projection fixes both, and the paper reports it also improves validation
    loss and downstream benchmarks on its own.
 2. SiTU-GLU (§2.3.2) as the expert activation — a SwiGLU whose two factors are
    both smoothly capped, so the product cannot blow up. See `situ_glu`.
 3. Quantile Balancing (§2.3.3) for load balancing — balancing ~10³ experts is
    outside the regime where the usual fixed-step bias update behaves. See
    `quantile_balancing_bias`.

MECHANICS (unchanged from the K2 / DeepSeek-V3 lineage)
-------------------------------------------------------
Routing is auxiliary-loss-FREE: a per-expert bias steers Top-k SELECTION only,
never the mixture weights or the router gradient. Dispatch uses the production
grouped-GEMM pattern — permute tokens so each expert's assignments are
contiguous, one `jax.lax.ragged_dot` per expert group, then un-permute and
weighted-sum. No token dropping, no capacity padding.

Pipeline per forward:
    1. Route:    sigmoid affinities (+ balancing bias on the SELECTION only)
                 -> Top-(k+1) -> the k routes taken, plus the cutoff QB needs
                 -> normalize the k gate weights from the TRUE scores.
    2. Down:     z = W↓ x                              (the latent, width ℓ)
    3. Dispatch: sort (token, expert) assignments by expert id; gather z rows
                 into expert-contiguous order; group_sizes = per-expert counts.
    4. Grouped GEMM: ragged_dot(z_sorted, W_in, group_sizes) -> SiTU-GLU ->
                 ragged_dot(a, W_out, group_sizes).   (gate+up fused into W_in)
    5. Combine:  scale rows by gate weight, scatter-add back to tokens -> u.
    6. Up:       y = shared(x) + W↑ RMSNorm(u).
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp

F32 = jnp.float32

# Xavier-uniform init with gain 2^{-2.5} (variance_scaling scale = gain² = 2^{-5}),
# replacing Flax NNX's default Linear kernel init. Biases stay at zero (the NNX
# default). The stacked expert weights below keep their own explicit fan-in init.
_XAVIER = nnx.initializers.variance_scaling(2**-5, "fan_avg", "uniform")


# --------------------------------------------------------------------------- #
#  SiTU-GLU (§2.3.2, Eq. 12) — the activation used by every expert.
# --------------------------------------------------------------------------- #
def situ_glu(
    gate_pre: jax.Array, up_pre: jax.Array, beta1: float = 4.0, beta2: float = 25.0
) -> jax.Array:
    """Sigmoid Tanh Unit GLU (K3 Eq. 12).

        SiTU-GLU(x) = [ β₁ tanh(W_g x / β₁) ⊙ Sigmoid(W_g x) ] ⊙ [ β₂ tanh(W_u x / β₂) ]

    Read it against SwiGLU, which is [ W_g x ⊙ Sigmoid(W_g x) ] ⊙ [ W_u x ]:
    each raw linear factor is replaced by the smooth cap softcap(t, β) = β·tanh(t/β).

    WHY. Both of SwiGLU's multiplicative factors are unbounded, so two large
    coordinates landing together produce an activation outlier — an overflow risk
    in low precision, and a source of the exploding activations §2.3 is fighting.
    The original GLU's plain sigmoid gate is bounded but loses Swish's roughly
    linear positive regime, which is where SwiGLU's empirical strength lives.
    The scaled tanh keeps both: β·tanh(t/β) = t + O(t³/β²), so near the origin
    SiTU-GLU matches SwiGLU to first order (and recovers it exactly as β → ∞),
    while |output| ≤ β₁β₂ = 100 everywhere (App. B, Eq. 18-19).

    K3 uses β₁ = 4 on the gate branch and β₂ = 25 on the up branch — the gate is
    capped tightly (it only has to modulate) and the value branch loosely.
    Unlike a hard clamp, the smooth cap keeps gradients nonzero away from the
    saturation boundary, which the paper reports trains better.
    """
    gate = beta1 * jnp.tanh(gate_pre / beta1) * jax.nn.sigmoid(gate_pre)
    up = beta2 * jnp.tanh(up_pre / beta2)
    return gate * up


class RMSNorm(nnx.Module):
    """RMSNorm with a learnable gain — used on the aggregated routed latent u
    (§2.3.1, the "Normalized" of Normalized LatentMoE)."""

    def __init__(self, dim: int, *, eps: float = 1e-5):
        self.eps = eps
        self.weight = nnx.Param(jnp.ones((dim,), F32))

    def __call__(self, x: jax.Array) -> jax.Array:
        xf = x.astype(F32)
        rms = jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + self.eps)
        return (xf * rms * self.weight[...]).astype(x.dtype)


# --------------------------------------------------------------------------- #
#  Dense FFN — the "Number of Dense Layers: 1" row of Table 1.
# --------------------------------------------------------------------------- #
class DenseFFN(nnx.Module):
    """Full-width SiTU-GLU FFN, no routing.

    K3 (like K2 and DeepSeek-V3) keeps the FIRST layer's channel mixer dense
    instead of MoE. The reason is a routing one: layer 1 sees representations
    that are still essentially the raw token embeddings, so its router would key
    on token identity and specialize experts by token rather than by function —
    a well-known source of early-training load collapse. One dense layer costs
    little and removes the problem.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        *,
        beta1: float = 4.0,
        beta2: float = 25.0,
        compute_dtype: jnp.dtype = F32,
        rngs: nnx.Rngs,
    ):
        self.beta1, self.beta2 = beta1, beta2
        self.compute_dtype = compute_dtype
        kg, ku, kd = jax.random.split(rngs.params(), 3)
        self.w_gate = nnx.Param(
            jax.random.normal(kg, (d_model, d_ff), F32) * (d_model**-0.5)
        )
        self.w_up = nnx.Param(
            jax.random.normal(ku, (d_model, d_ff), F32) * (d_model**-0.5)
        )
        self.w_down = nnx.Param(
            jax.random.normal(kd, (d_ff, d_model), F32) * (d_ff**-0.5)
        )

    def __call__(self, x: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Returns (y, aux) so it is drop-in interchangeable with StableLatentMoE.
        The aux dict is empty — a dense FFN has nothing to load-balance."""
        cd = self.compute_dtype
        xf = x.astype(cd)
        a = situ_glu(
            xf @ self.w_gate.astype(cd), xf @ self.w_up.astype(cd), self.beta1, self.beta2
        )
        return (a @ self.w_down.astype(cd)).astype(x.dtype), {}


# --------------------------------------------------------------------------- #
#  Stable LatentMoE
# --------------------------------------------------------------------------- #
class StableLatentMoE(nnx.Module):
    """Kimi K3 §2.3: token-dispatched grouped-GEMM MoE in a low-rank latent, with
    shared experts, normalized aggregation, SiTU-GLU, and Quantile Balancing.

    Args:
        d_model:     model width d.
        latent_dim:  routed-expert width ℓ (K3: 0.5·d). ℓ = d recovers a
                     conventional full-width MoE.
        d_ff:        per-ROUTED-expert hidden width (SiTU-GLU inner dim), in latent space.
        d_ff_shared: per-SHARED-expert hidden width (defaults to d_ff); shared
                     experts run at FULL width d_model in and out.
        n_routed:    number of routed experts E (K3: 896).
        n_shared:    number of always-on shared experts N_s (K3: 2), folded into
                     one wider SiTU-GLU.
        top_k:       experts activated per token (K3: 16).
        n_groups / topk_groups: optional group-limited ("node-limited") routing
                     from DeepSeek-V3 / K2. K3 does not describe it — Quantile
                     Balancing is its answer to routing at this scale — so this
                     defaults to OFF (n_groups=1). Kept because at real scale the
                     groups map to devices and it bounds all-to-all fan-out.
        beta1/beta2: SiTU-GLU soft caps (K3: 4 and 25).
        bias_balancing: enable the auxiliary-loss-free selection bias.
        aux_alpha:   coefficient of the optional Switch-style aux loss. K3 is
                     auxiliary-loss-FREE, so this defaults to 0.0; the term is
                     kept only as a comparison/diagnostic knob.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        latent_dim: int | None = None,
        n_routed: int = 896,
        n_shared: int = 2,
        top_k: int = 16,
        *,
        d_ff_shared: int | None = None,
        n_groups: int = 1,
        topk_groups: int = 1,
        norm_topk: bool = True,
        routed_scale: float = 1.0,
        beta1: float = 4.0,
        beta2: float = 25.0,
        bias_balancing: bool = True,
        aux_alpha: float = 0.0,
        rms_eps: float = 1e-5,
        compute_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        assert n_routed % n_groups == 0, "n_routed must be divisible by n_groups"
        assert 1 <= topk_groups <= n_groups, "need 1 <= topk_groups <= n_groups"
        # Quantile Balancing needs a Top-(k+1) cutoff, so there must be at least
        # one candidate expert beyond the k selected ones — inside the visible
        # groups when group-limited routing is on.
        assert top_k + 1 <= topk_groups * (n_routed // n_groups), (
            "top_k + 1 experts must fit inside the topk_groups selected groups "
            "(the +1 is the Quantile Balancing cutoff, Eq. 14)"
        )
        self.d_model = d_model
        self.latent = latent_dim if latent_dim is not None else d_model
        self.d_ff = d_ff
        self.E = n_routed
        self.top_k = top_k
        self.n_groups = n_groups
        self.topk_groups = topk_groups
        self.norm_topk = norm_topk
        self.routed_scale = routed_scale
        self.beta1, self.beta2 = beta1, beta2
        self.bias_balancing = bias_balancing
        self.aux_alpha = aux_alpha
        # Matmul dtype for the expert grouped GEMMs + shared experts (bf16 on H200).
        # Weights are stored fp32; the router, the combine (scatter-add), the
        # RMSNorm and the QB statistics stay fp32 for stable routing/balancing.
        self.compute_dtype = compute_dtype

        # --- Router (Eq. 13). Scores the FULL-WIDTH x, not the latent z: routing
        # is a decision about the token, and the down-projection is lossy. ---
        self.router = nnx.Linear(
            d_model, n_routed, use_bias=False, kernel_init=_XAVIER, rngs=rngs
        )
        # Selection bias b (Eq. 13/14). An nnx.Variable, not an nnx.Param: it is
        # never differentiated — the training loop overwrites it each step from
        # the QB update, and it is frozen at inference.
        self.router_bias = nnx.Variable(jnp.zeros((n_routed,), F32))

        # --- The latent bottleneck of Eq. 11: W↓ (d -> ℓ) and W↑ (ℓ -> d). ---
        self.w_down = nnx.Linear(
            d_model,
            self.latent,
            use_bias=False,
            kernel_init=_XAVIER,
            dtype=compute_dtype,
            param_dtype=F32,
            rngs=rngs,
        )
        # §2.3.1: RMSNorm sits BETWEEN expert aggregation and W↑, so W↑ always
        # sees a unit-scale input no matter which experts fired.
        self.u_norm = RMSNorm(self.latent, eps=rms_eps)
        self.w_up = nnx.Linear(
            self.latent,
            d_model,
            use_bias=False,
            kernel_init=_XAVIER,
            dtype=compute_dtype,
            param_dtype=F32,
            rngs=rngs,
        )

        # --- Routed experts E_i: R^ℓ -> R^ℓ, weights stacked over the expert axis
        # so the forward is two grouped GEMMs. Gate and up are fused into W_in,
        # so it is two rather than three. ---
        kin, kout = jax.random.split(rngs.params(), 2)
        self.w_in = nnx.Param(
            jax.random.normal(kin, (n_routed, self.latent, 2 * d_ff), F32)
            * (self.latent**-0.5)
        )
        self.w_out = nnx.Param(
            jax.random.normal(kout, (n_routed, d_ff, self.latent), F32) * (d_ff**-0.5)
        )

        # --- Shared experts E_j^shared: R^d -> R^d, at FULL width (that is the
        # point of the split: common transformations keep the whole channel
        # space). The N_s of them are folded into one wider SiTU-GLU, which is
        # arithmetically the same as summing N_s separate ones. ---
        sg, su, sd = jax.random.split(rngs.params(), 3)
        ish = (d_ff_shared if d_ff_shared is not None else d_ff) * n_shared
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
    def _route(self, x_flat: jax.Array):
        """Route tokens to experts (Eq. 13).

        Returns (top_idx [T,k], gate [T,k], logits [T,E], scores [T,E], cutoff [T]).

        Auxiliary-loss-free routing: the balancing bias is added ONLY to the
        SELECTION score and under stop_gradient, so it decides WHICH experts a
        token goes to but never appears in p_{i,j} nor in the router's gradient
        (Eq. 13: "Because b is omitted from p_{i,j}, it regulates dispatch
        without altering the mixture weights or the gradient-based optimization
        of the router").

        `cutoff` is the extra thing Quantile Balancing needs: selection runs
        Top-(k+1) instead of Top-k, and the (k+1)-th biased score α_i is exactly
        the bar an expert must clear to enter token i's Top-k. Getting it from
        the routing pass itself is what lets QB derive the next bias from a
        single forward with no separate token-side quantile (§2.3.3).
        """
        logits = self.router(x_flat).astype(F32)
        scores = jax.nn.sigmoid(logits)  # s_i = Sigmoid(W_r x_i)   [T,E]

        # Selection score s + b (bias only shifts WHO wins the Top-k).
        sel = scores + self.router_bias[...] if self.bias_balancing else scores
        sel = jax.lax.stop_gradient(sel)

        # Optional group-limited routing (OFF by default — see the class docstring).
        # Score each expert group by the sum of its top-2 selection scores, keep
        # the token's best `topk_groups` groups, mask the rest to -inf.
        if self.n_groups > 1:
            T = sel.shape[0]
            gsize = self.E // self.n_groups
            sel_g = sel.reshape(T, self.n_groups, gsize)
            top2, _ = jax.lax.top_k(sel_g, min(2, gsize))
            _, gidx = jax.lax.top_k(top2.sum(-1), self.topk_groups)
            keep = (
                jnp.zeros((T, self.n_groups), bool)
                .at[jnp.arange(T)[:, None], gidx]
                .set(True)
            )
            sel = jnp.where(jnp.repeat(keep, gsize, axis=-1), sel, -jnp.inf)

        # Top-(k+1): the first k entries are the routes taken, the last is the
        # per-token cutoff α_i used by the QB update.
        sel_top, idx_top = jax.lax.top_k(sel, self.top_k + 1)
        top_idx, cutoff = idx_top[:, : self.top_k], sel_top[:, self.top_k]

        # p_{i,j} = s_{i,j} / Σ_{r ∈ T_i} s_{i,r} — mixture weights from the TRUE
        # (un-biased) affinities of the selected experts, so the router keeps
        # exact gradients.
        gate = jnp.take_along_axis(scores, top_idx, axis=-1)
        if self.norm_topk:
            gate = gate / (gate.sum(-1, keepdims=True) + 1e-9)
        gate = gate * self.routed_scale

        return top_idx, gate, logits, scores, cutoff

    def _shared(self, x_flat: jax.Array) -> jax.Array:
        """Σ_j E_j^shared(x) — the N_s shared experts as one wider SiTU-GLU, at
        full model width. Runs in compute_dtype; the caller upcasts for the sum."""
        cd = self.compute_dtype
        xf = x_flat.astype(cd)
        a = situ_glu(
            xf @ self.ws_gate.astype(cd), xf @ self.ws_up.astype(cd), self.beta1, self.beta2
        )
        return a @ self.ws_down.astype(cd)

    # ----------------------------------------------------------------------- #
    def __call__(self, x: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
        """x: [B, L, d_model] -> (y, aux), implementing Eq. 11.

        aux carries what the training loop needs after the step:
            qb_bias     [E]  the NEXT router bias from Quantile Balancing (Eq. 14)
            group_sizes [E]  realized per-expert token counts (load diagnostics)
            load        [E]  the same as a fraction
            aux_loss    scalar, 0.0 unless aux_alpha > 0 (K3 is aux-loss-free)
        """
        B, L, d = x.shape
        T = B * L
        k = self.top_k
        xf = x.reshape(T, d)
        cd = self.compute_dtype

        top_idx, gate, router_logits, scores, cutoff = self._route(xf)

        # ---- down-project ONCE per token: only the ℓ-wide latent is dispatched.
        # This is the whole economics of LatentMoE — a token activating k experts
        # moves k copies of an ℓ-vector, not of a d-vector.
        z = self.w_down(xf).astype(cd)  # [T, ℓ]

        # ---- dispatch: flatten the (token, expert) assignments, sort by expert
        # id so each expert's rows are contiguous, and record the run lengths.
        flat_e = top_idx.reshape(T * k).astype(jnp.int32)  # expert per assignment
        flat_tok = jnp.repeat(jnp.arange(T, dtype=jnp.int32), k)  # token per assignment
        flat_w = gate.reshape(T * k).astype(F32)

        order = jnp.argsort(flat_e)
        sort_tok = flat_tok[order]
        sort_w = flat_w[order]
        group_sizes = jnp.bincount(flat_e, length=self.E)  # [E], sums to T*k

        z_sorted = z[sort_tok]  # [M, ℓ], M = T*k

        # ---- grouped GEMM: one matmul per expert over its contiguous rows.
        h = jax.lax.ragged_dot(z_sorted, self.w_in.astype(cd), group_sizes)
        g_, u_ = jnp.split(h, 2, axis=-1)  # [M, d_ff] each
        a = situ_glu(g_, u_, self.beta1, self.beta2)  # §2.3.2
        e_out = jax.lax.ragged_dot(a, self.w_out.astype(cd), group_sizes)  # [M, ℓ]

        # ---- combine: weight by p_i, un-permute, sum the top-k contributions.
        # This is u = Σ_{i ∈ T_k(x)} p_i E_i^routed(W↓x) of Eq. 11.
        e_out = e_out.astype(F32) * sort_w[:, None]
        u = jnp.zeros((T, self.latent), F32).at[sort_tok].add(e_out)

        # ---- Eq. 11's second line: y = Σ_j E_j^shared(x) + W↑ RMSNorm(u).
        # The RMSNorm (§2.3.1) is what makes the routed branch's scale
        # independent of which experts fired and how their weights fell.
        routed = self.w_up(self.u_norm(u).astype(cd)).astype(F32)
        out = (routed + self._shared(xf).astype(F32)).reshape(B, L, d).astype(cd)

        # ---- balancing statistics -------------------------------------------
        load = group_sizes.astype(F32) / (T * k)

        # Next step's bias, per Quantile Balancing (Eq. 14). Computed here
        # because it needs this batch's scores and cutoffs, but applied by the
        # training loop AFTER the step: "the update takes effect only in the next
        # step, i.e. a batch is never routed with a bias derived from itself".
        qb_bias = quantile_balancing_bias(scores, cutoff, k)

        # K3 is auxiliary-loss-free (§2.3.3), so aux_alpha = 0 by default and
        # this term vanishes. Kept for comparison with the Switch/DeepSeek-style
        # aux loss: E·<f_e, P_e> with f_e the realized load (constant) and P_e the
        # mean routing probability (where a gradient would flow).
        if self.aux_alpha:
            probs = jax.nn.softmax(router_logits, axis=-1).mean(0)
            aux_loss = self.aux_alpha * self.E * jnp.sum(load * probs)
        else:
            aux_loss = jnp.zeros((), F32)

        aux = {
            "load": load,
            "aux_loss": aux_loss,
            "group_sizes": group_sizes,
            "qb_bias": qb_bias,
        }
        return out, aux

    # ----------------------------------------------------------------------- #
    def dense_forward(self, x: jax.Array) -> jax.Array:
        """Reference path computing every expert densely (for tests only).
        Uses the SAME weights as __call__, so any mismatch is a dispatch/GEMM bug."""
        B, L, d = x.shape
        T = B * L
        xf = x.reshape(T, d)
        top_idx, gate, _, _, _ = self._route(xf)

        z = self.w_down(xf)
        full = (
            jnp.zeros((T, self.E), F32).at[jnp.arange(T)[:, None], top_idx].add(gate)
        )  # [T,E] sparse mixture weights
        h = jnp.einsum("tl,elf->tef", z, self.w_in[...])  # [T,E,2*d_ff]
        g_, u_ = jnp.split(h, 2, axis=-1)
        a = situ_glu(g_, u_, self.beta1, self.beta2)
        ye = jnp.einsum("tef,efl->tel", a, self.w_out[...])  # [T,E,ℓ]
        u = jnp.einsum("te,tel->tl", full, ye)
        out = self.w_up(self.u_norm(u)) + self._shared(xf)
        return out.reshape(B, L, d)


# --------------------------------------------------------------------------- #
#  Load balancing
# --------------------------------------------------------------------------- #
def quantile_balancing_bias(
    scores: jax.Array, cutoff: jax.Array, top_k: int
) -> jax.Array:
    """Quantile Balancing (K3 §2.3.3, Eq. 14) — the NEXT router bias, in closed form.

    THE PROBLEM. Auxiliary-loss-free balancing works by adding a per-expert bias
    b_j to the Top-k selection score. DeepSeek-V3 updates it with a fixed step,
    b_j ← b_j + γ·sign(target − load_j), where γ trades slow adaptation against
    oscillation. With 896 experts per layer that control loop is too crude:
    imbalance stalls expert-parallel training and can leave experts undertrained.

    THE IDEA. Ask directly what bias would give expert j its target load, and
    solve for it. For a batch of m tokens over n experts with Top-k selection the
    target is q = mk/n tokens per expert. Token i takes expert j iff j beats the
    token's own cutoff α_i (the (k+1)-th biased score, returned by `_route`):

        count_j(b̂_j) = Σ_i 1[ s_{i,j} + b̂_j > α_i ]

    which is monotonically decreasing in the threshold −b̂_j. Setting it equal to
    q therefore pins −b̂_j at the (q+1)-th largest MARGIN s_{i,j} − α_i; since
    q/m = k/n, that is the (1 − k/n)-quantile of expert j's margins:

        b̂_j^{t+1} ← −quantile_{1−k/n}( s_{:,j} − α^{(t)} )                Eq. 14
        b^{t+1}   ← b̂^{t+1} − mean(b̂^{t+1})·1

    The mean-subtraction removes a common offset, which cannot change Top-k
    selection (it shifts every expert equally) but keeps the biases from drifting.
    Note the old bias enters only through the cutoffs α — the margins use RAW
    scores — so there is no accumulating feedback and no step-size to tune; QB
    equilibrates in one step rather than converging over many.

    Args:
        scores: [T, E] raw sigmoid affinities s of this batch.
        cutoff: [T]    per-token Top-(k+1) biased score α.
        top_k:  k.
    Returns:
        [E] the bias to install BEFORE the next step (never this one — a batch
        must not be routed with a bias derived from itself).

    NOTE ON SCALE. This computes the exact quantile over the batch that is
    present. K3 cannot: its margins number in the millions and are spread across
    ranks and gradient-accumulation steps, so it estimates each expert's quantile
    from a HISTOGRAM of its margins — one all-reduce of per-rank bin counts, and
    the quantile read off the pooled counts. Because counts are additive, the
    estimate is the true whole-batch quantile up to the bin width, for a few
    hundred bins per expert (§2.3.3, "Histogram estimation").
    """
    E = scores.shape[-1]
    margins = jax.lax.stop_gradient(scores - cutoff[:, None])  # [T, E]
    b = -jnp.quantile(margins, 1.0 - top_k / E, axis=0)  # [E]
    return b - jnp.mean(b)


def update_router_bias(
    bias: jax.Array, group_sizes: jax.Array, lr: float = 1e-3
) -> jax.Array:
    """The PREVIOUS-generation fixed-step rule (DeepSeek-V3 / Kimi K2), kept for
    comparison with `quantile_balancing_bias`:

        b_j ← b_j + γ · sign(target_load − load_j)

    Nudges the bias up for under-loaded experts and down for over-loaded ones by a
    fixed step. Simple, but γ trades adaptation speed against load oscillation —
    the limitation §2.3.3 cites as its motivation for QB at 896 experts.
    """
    load = group_sizes.astype(F32) / jnp.sum(group_sizes).astype(F32)
    target = 1.0 / bias.shape[0]
    return bias + lr * jnp.sign(target - load)
