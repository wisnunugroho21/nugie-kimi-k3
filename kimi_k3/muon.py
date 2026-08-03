"""Per-Head Muon — §2.5 of the Kimi K3 report.

MUON, IN ONE PARAGRAPH.  For a matrix parameter, SGD-with-momentum applies an
update whose singular values can be wildly unequal: a few directions get almost
all the step, the rest get almost none. Muon fixes this by ORTHOGONALISING the
momentum matrix before applying it — replacing `M = U S V^T` with `U V^T`, so
every direction receives the same step size. The orthogonalisation is done with
a few Newton-Schulz iterations (matmuls only, no SVD). K2 adopted Muon; K3 keeps
it.

WHAT K3 CHANGES: THE "PER-HEAD" PART.  An attention projection such as W_q is
not really one matrix — it is `num_heads` independent matrices stacked into one
buffer. Orthogonalising the whole buffer treats all 96 heads as a single coupled
block, so:

    heads with large gradient/momentum scale dominate the shared update
    direction, while small-scale heads get updates that are not normalised
    enough.

So K3 partitions Q, K and V momentum along the head dimension and orthogonalises
each head's block separately. Each head then gets its own properly-scaled update.
The report reports more balanced learning dynamics across heads and better
stability at scale. It is also slightly *cheaper*: Newton-Schulz on tall thin
per-head blocks costs less than on the full projection.

WHAT IS NOT HERE.  §3.3 notes that K3 trains with "the weight-clipping mechanism
introduced in Kimi K2" (K2's QK-Clip / MuonClip) alongside Per-Head Muon. The K3
report does not restate that mechanism, so it is not reimplemented here.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax

from .layers import F32

# Quintic Newton-Schulz coefficients from the Muon reference implementation.
# The iteration X <- a X + b (X X^T) X + c (X X^T)^2 X drives the singular values
# of X toward 1 while only ever doing matmuls. These coefficients are tuned to
# converge fast rather than to be accurate: the result is approximately, not
# exactly, orthogonal, which is all the optimiser needs.
_NS_COEFFS = (3.4445, -4.7750, 2.0315)
_NS_STEPS = 5


def newton_schulz(x: jax.Array, steps: int = _NS_STEPS) -> jax.Array:
    """Approximate the orthogonal factor `U V^T` of `x = U S V^T`. x: [m, n]."""
    a, b, c = _NS_COEFFS
    transposed = x.shape[0] > x.shape[1]
    if transposed:  # iterate on the short side; the result transposes back
        x = x.T
    # Normalise so the spectral norm is <= 1, which is where the iteration
    # converges. (1e-7 guards an all-zero gradient.)
    x = x / (jnp.linalg.norm(x) + 1e-7)
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return x.T if transposed else x


def orthogonalize(m: jax.Array, num_heads: int = 0) -> jax.Array:
    """Newton-Schulz on `m`, per head if `num_heads > 0` (0 = whole matrix).

    Flax stores a Linear kernel as [in_features, out_features], and an attention
    projection's out_features is `num_heads * head_dim`. So the head partition is
    a split of the OUTPUT axis: [d_in, H*Dh] -> [H, d_in, Dh], one matrix per
    head, orthogonalised independently by vmap.
    """
    if not num_heads or m.ndim != 2:
        return newton_schulz(m)
    d_in, d_out = m.shape
    per_head = m.reshape(d_in, num_heads, d_out // num_heads).transpose(1, 0, 2)
    per_head = jax.vmap(newton_schulz)(per_head)
    return per_head.transpose(1, 0, 2).reshape(d_in, d_out)


class MuonState(NamedTuple):
    momentum: Any


def per_head_muon(
    momentum: float = 0.95,
    *,
    nesterov: bool = True,
    head_partition: Any = None,
) -> optax.GradientTransformation:
    """Muon as an optax transformation, with K3's per-head orthogonalisation.

    `head_partition` is a pytree matching the parameters, holding the head count
    for every attention projection and 0 ("orthogonalise the whole matrix")
    everywhere else. Build it with `head_partition_tree`. Note the sentinel is 0
    and not None: None is an empty node in a JAX pytree, not a leaf, so a tree
    of Nones would not line up with the gradient tree.

    The update is scaled by `0.2 * sqrt(max(m, n))`. Without it, an orthogonal
    update has RMS `1/sqrt(max(m,n))` per entry regardless of matrix size, so a
    single learning rate could not serve differently-shaped parameters; the
    factor puts Muon's update RMS on the same footing as AdamW's, which is what
    lets one LR schedule cover the whole model.

    SIGN CONVENTION — the thing to get right when chaining this.  Like
    `optax.scale_by_adam`, this transformation returns an update pointing ALONG
    the gradient, not against it. It is not a descent step on its own: the
    descent comes from `optax.scale_by_learning_rate`, which negates
    (`flip_sign=True` by default). So the chain must be

        per_head_muon() -> add_decayed_weights(wd) -> scale_by_learning_rate(lr)

    exactly as in the standard AdamW chain, and for the same reason:
    `add_decayed_weights` also assumes a gradient-direction update, so that the
    final negation turns both terms into descent at once. Negating here as well
    would cancel that negation and turn training into gradient ASCENT.
    """

    def init_fn(params):
        return MuonState(jax.tree.map(jnp.zeros_like, params))

    def update_fn(grads, state, params=None):
        del params
        heads = head_partition if head_partition is not None else jax.tree.map(lambda _: 0, grads)
        buf = jax.tree.map(lambda m, g: momentum * m + g, state.momentum, grads)
        raw = jax.tree.map(lambda m, g: g + momentum * m, buf, grads) if nesterov else buf

        def step(u, h):
            if u.ndim != 2:  # 1-D parameters are not Muon's business
                return u  # passed through in gradient direction, as below
            o = orthogonalize(u.astype(F32), int(h))
            # Gradient direction, NOT descent — see "SIGN CONVENTION" above.
            return 0.2 * (max(u.shape) ** 0.5) * o

        updates = jax.tree.map(step, raw, heads)
        return updates, MuonState(buf)

    return optax.GradientTransformation(init_fn, update_fn)


# ---------------------------------------------------------------------------
# Wiring it up
# ---------------------------------------------------------------------------

_ATTENTION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "q_up", "kv_up")


def head_partition_tree(params, num_heads: int):
    """Mark which parameters get per-head orthogonalisation.

    §2.5 names "the full Q, K, and V projection matrices". In this codebase that
    is KDA's `q_proj`/`k_proj`/`v_proj` and MLA's `q_up`/`kv_up` — the matrices
    whose output axis is `num_heads * head_dim`. Everything else (output
    projections, gates, MoE, embeddings) is left as whole-matrix Muon.
    """

    def label(path, leaf):
        keys = [str(getattr(p, "key", p)) for p in path]
        is_head_split = any(name in keys for name in _ATTENTION_PROJECTIONS)
        return num_heads if (is_head_split and leaf.ndim == 2) else 0

    return jax.tree_util.tree_map_with_path(label, params)


def kimi_k3_optimizer(
    learning_rate,
    num_heads: int,
    params,
    *,
    weight_decay: float = 0.1,
    momentum: float = 0.95,
    adam_b1: float = 0.9,
    adam_b2: float = 0.95,
) -> optax.GradientTransformation:
    """Per-Head Muon on matrices, AdamW on everything else.

    Muon is only defined for matrices. Embeddings, the LM head, norm scales and
    biases (all 1-D, or better treated as a lookup than a linear map) go to
    AdamW, which is the standard Muon recipe and the one K2/K3 follow.

    Weight decay is 0.1 throughout, per §3.3. The report also specifies a cosine
    schedule with a 1% linear warmup — pass that in as `learning_rate` (see
    `cosine_schedule_with_warmup`); the scaling-law study in §3.2 found cosine
    beat WSD once each schedule got its own hyper-parameter search.
    """
    labels = jax.tree.map(lambda p: "muon" if p.ndim == 2 else "adam", params)
    # `multi_transform` hands each branch the parameter tree with the *other*
    # branch's leaves replaced by `MaskedNode()`. The head-partition tree travels
    # alongside the gradients inside the Muon branch, so it has to be masked the
    # same way or the two trees will not line up.
    heads = jax.tree.map(
        lambda lbl, h: h if lbl == "muon" else optax.MaskedNode(),
        labels,
        head_partition_tree(params, num_heads),
    )
    return optax.multi_transform(
        {
            "muon": optax.chain(
                per_head_muon(momentum, head_partition=heads),
                optax.add_decayed_weights(weight_decay),
                optax.scale_by_learning_rate(learning_rate),
            ),
            "adam": optax.adamw(learning_rate, b1=adam_b1, b2=adam_b2, weight_decay=weight_decay),
        },
        labels,
    )


def cosine_schedule_with_warmup(peak_lr: float, total_steps: int, min_lr: float = 0.0):
    """§3.3: cosine decay with a 1% linear warmup."""
    warmup = max(1, int(0.01 * total_steps))
    return optax.join_schedules(
        [
            optax.linear_schedule(0.0, peak_lr, warmup),
            optax.cosine_decay_schedule(peak_lr, max(1, total_steps - warmup), alpha=min_lr / peak_lr),
        ],
        boundaries=[warmup],
    )
