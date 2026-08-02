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

from typing import NamedTuple

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
#  The SHARED branch of Eq. 11:  Σ_{j=1..N_s} E_j^shared(x)
# --------------------------------------------------------------------------- #
class SharedExperts(nnx.Module):
    """The always-on, FULL-WIDTH half of Stable LatentMoE (Eq. 11, first term).

    Every token goes through this path unconditionally — no router, no dispatch.
    That is the division of labour the LatentMoE split exists to create: shared
    experts "retain a full-width path for common transformations", while the
    routed experts specialize in a compact latent (see `RoutedExperts`). What is
    common to all tokens does not need to be selected, so it does not need to pay
    for routing or for the down-projection's information loss.

    K3 fixes N_s = 2 in every layer. The N_s experts are FOLDED INTO ONE wider
    SiTU-GLU: summing N_s separate GLUs of hidden width H is exactly one GLU of
    hidden width N_s·H, because the gate and up branches act per hidden unit and
    the down-projection is linear. One matmul triple instead of N_s.

    Shapes: d_model -> (n_shared · d_ff) -> d_model.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_shared: int = 2,
        *,
        beta1: float = 4.0,
        beta2: float = 25.0,
        compute_dtype: jnp.dtype = F32,
        rngs: nnx.Rngs,
    ):
        self.beta1, self.beta2 = beta1, beta2
        self.compute_dtype = compute_dtype
        self.inner = d_ff * n_shared

        kg, ku, kd = jax.random.split(rngs.params(), 3)
        self.w_gate = nnx.Param(
            jax.random.normal(kg, (d_model, self.inner), F32) * (d_model**-0.5)
        )
        self.w_up = nnx.Param(
            jax.random.normal(ku, (d_model, self.inner), F32) * (d_model**-0.5)
        )
        self.w_down = nnx.Param(
            jax.random.normal(kd, (self.inner, d_model), F32) * (self.inner**-0.5)
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """x: [..., d_model] -> [..., d_model], in compute_dtype.

        Left in compute_dtype rather than upcast: the caller decides the
        precision of the sum it feeds into (StableLatentMoE adds this to the
        routed branch in fp32)."""
        cd = self.compute_dtype
        xf = x.astype(cd)
        a = situ_glu(
            xf @ self.w_gate.astype(cd), xf @ self.w_up.astype(cd), self.beta1, self.beta2
        )
        return a @ self.w_down.astype(cd)


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

    Structurally this IS a shared expert — an always-on full-width SiTU-GLU — so
    it wraps `SharedExperts` with n_shared = 1 rather than duplicating it. The
    only thing it adds is the (y, aux) return signature that makes it drop-in
    interchangeable with StableLatentMoE as a channel mixer.
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
        self.ffn = SharedExperts(
            d_model,
            d_ff,
            n_shared=1,
            beta1=beta1,
            beta2=beta2,
            compute_dtype=compute_dtype,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Returns (y, aux). The aux dict is empty — a dense FFN has nothing to
        load-balance."""
        return self.ffn(x).astype(x.dtype), {}


# --------------------------------------------------------------------------- #
#  The ROUTED branch of Eq. 11:  W↑ RMSNorm( Σ_{i ∈ T_k(x)} p_i E_i^routed(W↓x) )
# --------------------------------------------------------------------------- #
#  Routing (§2.3.3, Eq. 13) — the assignment, kept apart from the experts.
# --------------------------------------------------------------------------- #
class Routing(NamedTuple):
    """One batch's routing decision — the complete output of Eq. 13.

    Named rather than a bare tuple because three of these five fields exist for
    reasons that are easy to mix up:

      top_idx  [T, k]  T_i: which experts each token was assigned to.
      gate     [T, k]  p_{i,j}: the mixture weights for those experts. Derived
                       from the TRUE affinities, never from the biased ones.
      logits   [T, E]  the raw router output, kept only so the optional
                       Switch-style aux loss can reuse it without a second matmul.
      scores   [T, E]  s = Sigmoid(logits), the true affinities over all experts.
                       Quantile Balancing needs the full row, not just the top-k.
      cutoff   [T]     α_i: the (k+1)-th BIASED score — the bar an expert had to
                       clear to enter this token's top-k. Only QB uses it.
    """

    top_idx: jax.Array
    gate: jax.Array
    logits: jax.Array
    scores: jax.Array
    cutoff: jax.Array


class Router(nnx.Module):
    """K3 Eq. 13 — who goes where, and nothing else.

    Deliberately separate from `RoutedExperts`: this class decides the
    ASSIGNMENT (which experts a token visits and with what weight), while the
    expert bank decides WHAT HAPPENS once it gets there. They are coupled only
    through a `Routing`, which makes both halves testable on their own and makes
    the load-balancing machinery — which is entirely a routing concern — sit
    where it belongs.

    Everything about balancing lives here too:
      * the selection bias b (Eq. 13), an nnx.Variable rather than an nnx.Param
        because it is never differentiated;
      * `balance()`, the Quantile Balancing update that produces the NEXT bias
        (Eq. 14);
      * `aux_loss()`, the auxiliary loss K3 does NOT use, kept for comparison.

    Args:
        d_model:     model width. The router scores the FULL-WIDTH token, not the
                     latent — routing is a decision about the token, and the
                     down-projection into the expert latent is lossy.
        n_routed:    number of experts E (K3: 896).
        top_k:       experts activated per token (K3: 16).
        n_groups / topk_groups: optional group-limited ("node-limited") routing
                     from DeepSeek-V3 / K2. K3 does not describe it — Quantile
                     Balancing is its answer to routing at this scale — so this
                     defaults to OFF (n_groups=1). Kept because at real scale the
                     groups map to devices and it bounds all-to-all fan-out.
        norm_topk:   renormalize the k gate weights to sum to 1.
        routed_scale: extra scale on the normalized weights (DeepSeek's
                     routed_scaling_factor).
        bias_balancing: enable the auxiliary-loss-free selection bias.
        aux_alpha:   coefficient of the optional Switch-style aux loss. K3 is
                     auxiliary-loss-FREE, so this defaults to 0.0.
    """

    def __init__(
        self,
        d_model: int,
        n_routed: int = 896,
        top_k: int = 16,
        *,
        n_groups: int = 1,
        topk_groups: int = 1,
        norm_topk: bool = True,
        routed_scale: float = 1.0,
        bias_balancing: bool = True,
        aux_alpha: float = 0.0,
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
        self.E = n_routed
        self.top_k = top_k
        self.n_groups = n_groups
        self.topk_groups = topk_groups
        self.norm_topk = norm_topk
        self.routed_scale = routed_scale
        self.bias_balancing = bias_balancing
        self.aux_alpha = aux_alpha

        # Scoring stays in fp32 (no compute_dtype): the top-k boundary and the
        # balancing quantiles are decisions, and a bf16 tie is a different route.
        self.linear = nnx.Linear(
            d_model, n_routed, use_bias=False, kernel_init=_XAVIER, rngs=rngs
        )
        # Selection bias b (Eq. 13/14). An nnx.Variable, not an nnx.Param: it is
        # never differentiated — the training loop overwrites it each step from
        # the QB update, and it is frozen at inference.
        self.bias = nnx.Variable(jnp.zeros((n_routed,), F32))

    def __call__(self, x_flat: jax.Array) -> Routing:
        """x_flat: [T, d_model] -> Routing (Eq. 13).

        Auxiliary-loss-free routing: the balancing bias is added ONLY to the
        SELECTION score and under stop_gradient, so it decides WHICH experts a
        token goes to but never appears in p_{i,j} nor in the router's gradient
        (Eq. 13: "Because b is omitted from p_{i,j}, it regulates dispatch
        without altering the mixture weights or the gradient-based optimization
        of the router").

        Selection runs Top-(k+1) rather than Top-k. The extra entry is the
        per-token cutoff α_i, and taking it from the routing pass itself is what
        lets Quantile Balancing derive the next bias from a single forward with
        no separate token-side quantile (§2.3.3).
        """
        logits = self.linear(x_flat).astype(F32)
        scores = jax.nn.sigmoid(logits)  # s_i = Sigmoid(W_r x_i)   [T,E]

        # Selection score s + b (bias only shifts WHO wins the Top-k).
        sel = scores + self.bias[...] if self.bias_balancing else scores
        sel = jax.lax.stop_gradient(sel)

        # Optional group-limited routing (OFF by default — see the class
        # docstring). Score each expert group by the sum of its top-2 selection
        # scores, keep the token's best `topk_groups` groups, mask the rest.
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

        sel_top, idx_top = jax.lax.top_k(sel, self.top_k + 1)
        top_idx, cutoff = idx_top[:, : self.top_k], sel_top[:, self.top_k]

        # p_{i,j} = s_{i,j} / Σ_{r ∈ T_i} s_{i,r} — mixture weights from the TRUE
        # (un-biased) affinities of the selected experts, so the router keeps
        # exact gradients.
        gate = jnp.take_along_axis(scores, top_idx, axis=-1)
        if self.norm_topk:
            gate = gate / (gate.sum(-1, keepdims=True) + 1e-9)
        gate = gate * self.routed_scale

        return Routing(top_idx, gate, logits, scores, cutoff)

    def balance(self, routing: Routing) -> jax.Array:
        """The NEXT selection bias, per Quantile Balancing (Eq. 14) -> [E].

        Computed from this batch but applied by the training loop AFTER the step:
        "the update takes effect only in the next step, i.e. a batch is never
        routed with a bias derived from itself" (§2.3.3). See
        `quantile_balancing_bias` for the derivation.
        """
        return quantile_balancing_bias(routing.scores, routing.cutoff, self.top_k)

    def aux_loss(self, routing: Routing, load: jax.Array) -> jax.Array:
        """The Switch/DeepSeek-style auxiliary load-balancing loss.

        K3 is auxiliary-loss-FREE (§2.3.3), so `aux_alpha` is 0 by default and
        this returns exactly zero. Kept as a comparison knob: E·<f_e, P_e> with
        f_e the realized load (a constant, no gradient) and P_e the mean routing
        probability, which is where a gradient would flow. It reuses the logits
        already in `routing`, so enabling it costs no second router matmul.
        """
        if not self.aux_alpha:
            return jnp.zeros((), F32)
        probs = jax.nn.softmax(routing.logits, axis=-1).mean(0)
        return self.aux_alpha * self.E * jnp.sum(load * probs)


# --------------------------------------------------------------------------- #
#  The ROUTED branch of Eq. 11:  W↑ RMSNorm( Σ_{i∈T_k(x)} p_i E_i(W↓x) )
# --------------------------------------------------------------------------- #
class RoutedExperts(nnx.Module):
    """The sparse, LATENT-WIDTH half of Stable LatentMoE (Eq. 11, second term).

    Given a `Routing` — produced by `Router`, not by this class — it applies the
    selected experts and returns the branch's contribution at model width. It
    owns the latent bottleneck W↓/W↑, the stacked expert bank, the
    dispatch/grouped-GEMM/combine machinery, and the §2.3.1 RMSNorm; it owns no
    part of the routing DECISION.

    THE LATENT IS THE POINT. In a conventional MoE each selected expert receives
    the full d-dimensional token, so communication and expert-weight traffic grow
    with how many experts a token activates — and K3 wants 16 of them. Here the
    token is down-projected ONCE to width ℓ (K3: 0.5·d) and only that ℓ-vector is
    dispatched, so raising top_k no longer raises the per-token traffic
    proportionally in d. Shared experts (see `SharedExperts`) keep the full-width
    path, so nothing is lost overall — the two branches divide the work.

    Args:
        d_model:    model width d, the width W↓/W↑ bridge to and from.
        d_ff:       per-expert hidden width (SiTU-GLU inner dim), in latent space.
        latent_dim: routed-expert width ℓ. ℓ = d recovers a conventional MoE.
        n_routed:   number of experts E (K3: 896). Must match the router's.
        beta1/beta2: SiTU-GLU soft caps (K3: 4 and 25).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        latent_dim: int | None = None,
        n_routed: int = 896,
        *,
        beta1: float = 4.0,
        beta2: float = 25.0,
        rms_eps: float = 1e-5,
        compute_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.d_model = d_model
        self.latent = latent_dim if latent_dim is not None else d_model
        self.d_ff = d_ff
        self.E = n_routed
        self.beta1, self.beta2 = beta1, beta2
        # Matmul dtype for the expert grouped GEMMs (bf16 on H200). Weights are
        # stored fp32; the combine (scatter-add) and the RMSNorm stay fp32.
        self.compute_dtype = compute_dtype

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

        # --- Expert bank E_i: R^ℓ -> R^ℓ, weights stacked over the expert axis
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

    # ----------------------------------------------------------------------- #
    def __call__(
        self, x_flat: jax.Array, routing: Routing
    ) -> tuple[jax.Array, jax.Array]:
        """x_flat: [T, d_model], routing from `Router` -> (y [T, d_model] fp32,
        group_sizes [E]).

        Computes W↑ RMSNorm(u) with u = Σ_{i ∈ T_k(x)} p_i E_i(W↓x), i.e. the
        routed half of Eq. 11 including its up-projection back to model width —
        so the caller only has to add the shared branch.

        `group_sizes` (per-expert token counts) falls out of the dispatch, which
        needs it anyway; it is returned because the load diagnostics and the
        optional aux loss are computed from it.
        """
        T = x_flat.shape[0]
        k = routing.top_idx.shape[1]  # taken from the routing, not re-configured
        cd = self.compute_dtype

        # ---- down-project ONCE per token: only the ℓ-wide latent is dispatched.
        # This is the whole economics of LatentMoE — a token activating k experts
        # moves k copies of an ℓ-vector, not of a d-vector.
        z = self.w_down(x_flat).astype(cd)  # [T, ℓ]

        # ---- dispatch: flatten the (token, expert) assignments, sort by expert
        # id so each expert's rows are contiguous, and record the run lengths.
        flat_e = routing.top_idx.reshape(T * k).astype(jnp.int32)
        flat_tok = jnp.repeat(jnp.arange(T, dtype=jnp.int32), k)
        flat_w = routing.gate.reshape(T * k).astype(F32)

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

        # ---- W↑ RMSNorm(u). The RMSNorm (§2.3.1) is what makes this branch's
        # scale independent of which experts fired and how their weights fell,
        # so it is comparable with the full-width shared branch it is added to.
        y = self.w_up(self.u_norm(u).astype(cd)).astype(F32)
        return y, group_sizes

    def dense_forward(self, x_flat: jax.Array, routing: Routing) -> jax.Array:
        """Reference path computing every expert densely (for tests only).
        Uses the SAME weights as __call__, so any mismatch is a dispatch/GEMM bug."""
        T = x_flat.shape[0]
        z = self.w_down(x_flat)
        full = (
            jnp.zeros((T, self.E), F32)
            .at[jnp.arange(T)[:, None], routing.top_idx]
            .add(routing.gate)
        )  # [T,E] sparse mixture weights
        h = jnp.einsum("tl,elf->tef", z, self.w_in[...])  # [T,E,2*d_ff]
        g_, u_ = jnp.split(h, 2, axis=-1)
        a = situ_glu(g_, u_, self.beta1, self.beta2)
        ye = jnp.einsum("tef,efl->tel", a, self.w_out[...])  # [T,E,ℓ]
        u = jnp.einsum("te,tel->tl", full, ye)
        return self.w_up(self.u_norm(u))


# --------------------------------------------------------------------------- #
#  Stable LatentMoE = router + routed branch + shared branch
# --------------------------------------------------------------------------- #
class StableLatentMoE(nnx.Module):
    """Kimi K3 §2.3, Eq. 11 — the channel mixer, assembled from three pieces:

        y = Σ_{j=1..N_s} E_j^shared(x)   +   W↑ RMSNorm( Σ_{i∈T_k(x)} p_i E_i^routed(W↓x) )
            └────── self.shared ──────┘       └───────────── self.routed ─────────────┘
                                                    with T_k, p from self.router

    The shared/routed split is the paper's central idea: the two branches differ
    in EVERY dimension that matters — the shared one is dense, always-on, and
    full-width; the routed one is sparse, selected, and confined to a latent of
    width ℓ. That asymmetry is what lets K3 hold 896 experts and activate 16 per
    token without the communication cost a conventional MoE would pay.

    The router is a third, separate piece because assignment and computation are
    genuinely different concerns: `Router` answers "which experts, how strongly"
    (Eq. 13, plus all of load balancing), `RoutedExperts` answers "what those
    experts do". This class just wires them together and assembles the
    diagnostics.

    Args:
        d_model:     model width d.
        latent_dim:  routed-expert width ℓ (K3: 0.5·d). ℓ = d recovers a
                     conventional full-width MoE.
        d_ff:        per-ROUTED-expert hidden width, in latent space.
        d_ff_shared: per-SHARED-expert hidden width (defaults to d_ff); shared
                     experts run at FULL width d_model in and out.
        n_routed:    number of routed experts E (K3: 896).
        n_shared:    number of always-on shared experts N_s (K3: 2).
        top_k:       experts activated per token (K3: 16).
        Remaining keywords are forwarded to `Router` — see it for group-limited
        candidates and load-balancing options.
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
        self.d_model = d_model
        self.compute_dtype = compute_dtype

        self.router = Router(
            d_model,
            n_routed=n_routed,
            top_k=top_k,
            n_groups=n_groups,
            topk_groups=topk_groups,
            norm_topk=norm_topk,
            routed_scale=routed_scale,
            bias_balancing=bias_balancing,
            aux_alpha=aux_alpha,
            rngs=rngs,
        )
        self.shared = SharedExperts(
            d_model,
            d_ff_shared if d_ff_shared is not None else d_ff,
            n_shared=n_shared,
            beta1=beta1,
            beta2=beta2,
            compute_dtype=compute_dtype,
            rngs=rngs,
        )
        self.routed = RoutedExperts(
            d_model,
            d_ff,
            latent_dim=latent_dim,
            n_routed=n_routed,
            beta1=beta1,
            beta2=beta2,
            rms_eps=rms_eps,
            compute_dtype=compute_dtype,
            rngs=rngs,
        )

    @property
    def router_bias(self) -> nnx.Variable:
        """Forwarder to `self.router.bias` (Eq. 13's selection bias b), so callers
        such as `apply_quantile_balancing` need not know which sub-module holds
        it. Returns the Variable itself, so `moe.router_bias[...] = b` writes
        through."""
        return self.router.bias

    def __call__(self, x: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
        """x: [B, L, d_model] -> (y, aux), i.e. Eq. 11 over a flattened batch.

        aux carries what the training loop needs after the step:
            qb_bias     [E]  the NEXT router bias from Quantile Balancing (Eq. 14)
            group_sizes [E]  realized per-expert token counts (load diagnostics)
            load        [E]  the same as a fraction
            aux_loss    scalar, 0.0 unless aux_alpha > 0 (K3 is aux-loss-free)
        """
        B, L, d = x.shape
        xf = x.reshape(B * L, d)

        routing = self.router(xf)  # Eq. 13: who goes where
        routed, group_sizes = self.routed(xf, routing)  # Eq. 11's second term
        out = routed + self.shared(xf).astype(F32)  # + the first term

        load = group_sizes.astype(F32) / (B * L * self.router.top_k)
        aux = {
            "load": load,
            "aux_loss": self.router.aux_loss(routing, load),
            "group_sizes": group_sizes,
            "qb_bias": self.router.balance(routing),
        }
        return out.reshape(B, L, d).astype(self.compute_dtype), aux

    def dense_forward(self, x: jax.Array) -> jax.Array:
        """Reference path computing every routed expert densely (for tests only).
        Uses the SAME weights as __call__, so any mismatch is a dispatch/GEMM bug."""
        B, L, d = x.shape
        xf = x.reshape(B * L, d)
        routing = self.router(xf)
        out = self.routed.dense_forward(xf, routing) + self.shared(xf)
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
