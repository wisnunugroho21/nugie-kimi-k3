"""Gated Delta Rule-2 — training-time recurrence cores in JAX.

Implements the token mixer of Hatamizadeh, Choi, Kautz, "Gated DeltaNet-2:
Decoupling Erase and Write in Linear Attention" (arXiv:2605.22791).
Equation numbers below are cited as "main text / Appendix A" where the
equation appears in both.

The state S ∈ R^{dk×dv} is an associative memory mapping key directions to
value rows; the output reads it with the query, o_t = S_tᵀ q_t (Eq. 1).
Each token applies three edits (Eq. 10/29):

    S_r = (I − k_r e_rᵀ) Diag(α_r) S_{r−1} + k_r z_rᵀ,
    e_r = b_r ⊙ k_r,   z_r = w_r ⊙ v_r,   α_r = exp(g_r) ∈ (0,1]

  1. FORGET: Diag(α_r) shrinks each key-channel row (passive decay).
  2. ERASE:  −k_r e_rᵀ S removes what the memory returns along the gated
     read direction e_r; the erase gate b picks WHICH key channels.
  3. WRITE:  +k_r z_rᵀ inserts the gated value; the write gate w picks
     WHICH value channels.
The "-2" is the decoupling: classic (Gated) DeltaNet ties erase and write
to one scalar β_r; here b (key side) and w (value side) are independent
per-channel gates.

Six implementations of the same recurrence:

  _recurrent_single           token-by-token scan of Eq. 9/29. O(L·dk·dv),
                              trivially correct — the verification oracle.
  _chunkwise_single_faithful  chunkwise WY form exactly as printed in the
                              paper (Eqs. 18-25 / 30-44), including the
                              explicit inverse A = (I + T)^{-1}. Overflows
                              fp32 when a chunk's cumulative log-decay G
                              drops below ≈ −88 (it materializes exp(−G)).
  _chunkwise_single_stacked_RHS_solve
                              faithful + the solver optimization: Y and U
                              come from ONE triangular solve with stacked
                              RHS [Ē|Z] instead of the explicit inverse.
                              Same literal factors, same ≈ −88 overflow
                              limit — the baseline the optimized cores
                              below build on.
  _chunkwise_single_centered  same algebra with per-chunk exponent
                              centering: all within-chunk exponents are
                              shifted by c = G_C/2, which cancels exactly
                              in every product (they only consume
                              differences of G) and is re-attached against
                              S0 via diag(exp(c)). Doubles the safe range
                              to |G_C| ≈ 176; used by the public entry point.
  _chunkwise_single_pairwise  overflow-proof fallback: forms every decay
                              factor directly as exp(G_r − G_s) per
                              (r, s, channel) triple — all exponents ≤ 0,
                              so NO range limit — at the cost of a
                              [N, C, C, dk] intermediate (×C memory) and
                              einsums instead of matmuls for T and A_qk.
                              Use when gate values exceed the centered
                              core's range and chunk_size cannot shrink.
  _chunkwise_single_subchunking
                              middle ground (GLA-style secondary chunking):
                              T and A_qk are assembled from sub-blocks of
                              size c. Off-diagonal block pairs factor the
                              decay through the row-block's entry boundary
                              — both factors have exponents ≤ 0 — and stay
                              batched MATMULS; only the c×c diagonal blocks
                              use the pairwise form. NO range limit, with
                              memory overhead ×(C/c) instead of ×C.

Chunkwise idea (Sec. 3.3 / App. A): substituting S_r = Diag(γ_r) Ŝ_r with
γ_r = exp(Σ_{i≤r} g_i) removes the decay from the recurrence — Ŝ follows a
PURE delta rule whose C-step unroll is a product of rank-one corrections.
Stacking its residual rows ρ_r = z_r − ē_rᵀ Ŝ_{r−1} gives one unit
lower-triangular system (I + T) R = Z − Ē S0 (Eq. 39), solved by forward
substitution; the output and end-of-chunk state are then dense matmuls.
Only the cross-chunk state remains sequential.

Chunkwise WY form (Eqs. 18-25 / 30-44), written in the paper's literal
factors — the reference realization that _chunkwise_single_faithful
implements verbatim; the other cores compute the same quantities with
safer factorizations (see the strategy notes below):
    G_r     = cumsum(g)             (inclusive, within chunk)       Eq. 18/30
    gamma   = exp(G)                                                Eq. 18/30
    gamma_C = gamma[-1]             (total chunk decay)             Eq. 18/30
    Kbar    = exp(-G) ⊙ K           (the overflow source)           Eq. 19/33
    Ebar    = gamma ⊙ (B ⊙ K)       (decay-absorbed erase factor)   Eq. 20/33
    Z       = W ⊙ V                 (gated write targets)           Eq. 20/33
    Qgamma  = gamma ⊙ Q             (decay-weighted queries)        Eq. 24/43
    T       = tril(Ebar Kbar^T, -1)                                 Eq. 21/34
    A       = (I + T)^{-1}          (unit lower-triangular solve)   Eq. 21/34
    Y       = A Ebar                (erase-side WY auxiliary)       Eq. 22/34
    U       = A Z                   (write-side WY auxiliary)       Eq. 22/34
    R       = U - Y S0              (the chunk's residual writes)   Eq. 35
    Aqk     = tril(Qgamma Kbar^T)                                   Eq. 25/43
    O       = Qgamma S0 + Aqk R                                     Eq. 24/44
    Ktail   = exp(G_C - G) ⊙ K      (tail-decayed keys)             Eq. 23/41
    S_C     = diag(gamma_C) S0 + Ktail^T R                          Eq. 23/40

--------- Optimization strategies ---------

Two independent optimizations distinguish the chunkwise cores.

(a) SOLVER. Y and U solve the SAME unit lower-triangular system
    (I + T) X = RHS with right-hand sides Ē and Z (Eqs. 45/46). The
    faithful core materializes the explicit inverse A = (I + T)^{-1} and
    multiplies twice, as the paper writes it; every other chunkwise core
    uses ONE forward substitution over the stacked RHS [Ē | Z] (_wy_solve)
    — fewer flops (a (dk+dv)-RHS solve replaces a C-RHS solve plus two
    matmuls), one less [C, C] intermediate, and better numerics than
    multiplying by an explicitly formed inverse.

(b) DECAY FACTORIZATION. Define the pairwise decay ratios

        D_rsc = exp(G_rc - G_sc) · 1[s ≤ r]   (pair (r, s), key channel c)

    — the only decay factors T and A_qk actually consume. Together with
    the tail/carry ratios exp(G_C − G_r) and exp(G_C), every one of them
    has exponent ≤ 0, so the exact math is overflow-free; overflow can
    only enter through HOW a core factorizes D_rsc into materialized
    tensors. The optimized cores differ in that choice:

_chunkwise_single_stacked_RHS_solve — (a) only; keeps the paper's literal
factors, so the ≈ 88 overflow limit remains:
    D_rsc  = exp(G_rc)·exp(-G_sc)     (the exp(-G) half still overflows)
    [Y|U]  = (I + T)^{-1} [Ē|Z]       (one solve, stacked RHS)

_chunkwise_single_centered — (a) + the faithful exp(G_r)·exp(-G_s) split
with exponents shifted by c = G_C/2 per channel (matmuls; range doubles
to |G_C| ~ 176):
    Gc     = G - G_C/2            (spans ±|G_C|/2, not [G_C, 0])
    delta  = exp(G_C/2) <= 1      (re-attaches the shift to S0)
    Kbar   = exp(-Gc) ⊙ K,   Ebar_c = exp(Gc) ⊙ (B ⊙ K),
    Qg_c   = exp(Gc) ⊙ Q
    D_rsc  = exp(Gc_rc)·exp(-Gc_sc) = exp(G_rc - G_sc)  (c cancels)
    T      = tril(Ebar_c Kbar^T, -1),   Aqk = tril(Qg_c Kbar^T)
    R      = U - Y (delta ⊙ S0),  O = Qg_c (delta ⊙ S0) + Aqk R
             (delta ⊙ S0 restores Y S0 and Qgamma S0 exactly)      Eq. 35, 24/44

_chunkwise_single_pairwise — (a) + every triple formed directly (einsums
over a [N, C, C, dk] tensor; no range limit, x C memory):
    D_rsc  = exp(G_rc - G_sc)·1[s≤r]   (mask BEFORE exp: anti-causal
             exponents are ≥ 0 and would give inf·0 = NaN after)
    T_rs   = Σ_c (B⊙K)_rc K_sc D_rsc,  s < r                       Eq. 21/34
    Aqk_rs = Σ_c Q_rc K_sc D_rsc,      s ≤ r                       Eq. 25/43
    Ebar, Qgamma, Ktail as in the shared form (exponents ≤ 0, safe)

_chunkwise_single_subchunking — (a) + M = C/c sub-blocks, decay factored
through each row-block's entry boundary B_i = G at the last position
before block i (matmul-dominated; no range limit, x C/c memory):
    off-diagonal (r in block i, s in block j < i):
        D_rsc = exp(G_rc - B_ic)·exp(B_ic - G_sc)   (s ≤ bnd ≤ r, so
                BOTH exponents ≤ 0; batched matmuls per row-block)
    diagonal (r, s in the same block):
        Grel  = G - B_i ≤ 0        (block-local log-decay)
        D_rsc = exp(Grel_rc - Grel_sc)·1[s≤r]   (pairwise form, but
                only over c×c blocks; B_i cancels in the difference)
    T, Aqk = off-diagonal blocks + scattered diagonal blocks       Eq. 21/34, 25/43

The hand-derived backward of Appendix B is intentionally not implemented:
jax.grad differentiates through solve_triangular and the elementwise gate
products and reconstructs exactly those vector-Jacobian products. A manual
backward is only needed for a fused Triton/Pallas kernel.

-------------------------------------------

Shape conventions (one head): q, k, g, b: [L, dk]; v, w: [L, dv];
S0: [dk, dv]. Public entry points add leading [B, H] axes via vmap.
All math runs in fp32 (paper App. D). Every core is verified against
_recurrent_single and an independent float64 oracle in tests/test_rule.py.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

# Compute dtype for all internal math (paper App. D: gate/decay math in fp32).
D_TYPE = jnp.float32


def _recurrent_single(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    b: jax.Array,
    w: jax.Array,
    S0: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Token-by-token reference forward for ONE (batch, head) pair.

    Direct scan of Eq. 9 (three-line factored form, algebraically equal to
    the (I − k e ᵀ) Diag(α) form of Eq. 10/29). O(L·dk·dv), no triangular
    solve — a trustworthy oracle for verifying the chunkwise paths, and the
    form used at inference (one token at a time).

    Args:
      q, k, g, b: [L, dk]   v, w: [L, dv]   S0: [dk, dv]
    Returns:
      (O: [L, dv], S_final: [dk, dv])
    """
    q = q.astype(D_TYPE)
    k = k.astype(D_TYPE)
    v = v.astype(D_TYPE)
    g = g.astype(D_TYPE)
    b = b.astype(D_TYPE)
    w = w.astype(D_TYPE)
    S0 = S0.astype(D_TYPE)

    alpha = jnp.exp(g)  # Eq. 12/30: α = exp(g), g ≤ 0 so α ∈ (0,1]  [L, dk]
    e = b * k  # Eq. 8: e = b ⊙ k — erase gate picks key channels    [L, dk]
    z = w * v  # Eq. 8: z = w ⊙ v — write gate picks value channels  [L, dv]

    def step(S, inp):
        # S: [dk, dv]; per-token slices qt, kt, at, et: [dk], zt: [dv].
        qt, kt, at, et, zt = inp

        # Column vectors for the rank-one linear algebra below.
        qt = qt[:, None]  # [dk, 1]
        kt = kt[:, None]  # [dk, 1]
        at = at[:, None]  # [dk, 1]
        et = et[:, None]  # [dk, 1]
        zt = zt[:, None]  # [dv, 1]

        # Eq. 9:  S̄_t = Diag(α_t) S_{t−1} — passive forgetting: each
        # key-channel row shrinks by its own α ∈ (0,1].            [dk, dv]
        S_bar = at * S

        # Eq. 9:  r_t = S̄_tᵀ e_t — RECALL: what the decayed memory returns
        # along the gated erase direction. With b = 1 this is the classic
        # delta-rule read S̄ᵀk; b < 1 mutes key channels, protecting their
        # stored content from the upcoming subtraction.             [dv, 1]
        r_t = S_bar.T @ et

        # Eq. 9/15:  S_t = S̄_t + k_t (z_t − r_t)ᵀ — delta write: store only
        # the RESIDUAL between the gated target z_t and the recalled r_t, at
        # key k_t. If memory already holds the target, nothing is written
        # (no unbounded accumulation, unlike vanilla linear attention).
        # Also one gradient step on ½‖Sᵀk_t − target‖² (fast-weight view,
        # Eqs. 13-15).                                             [dk, dv]
        S_new = S_bar + kt * (zt - r_t).T

        # Eq. 1:  o_t = S_tᵀ q_t — read the post-update memory.     [dv, 1]
        o_t = S_new.T @ qt

        return S_new, o_t

    # o stacks per-token outputs: [L, dv, 1] -> squeeze to [L, dv].
    S_final, o = lax.scan(step, S0, (q, k, alpha, e, z))
    return o.squeeze(-1), S_final


# --------------------------------------------------------------------------- #
#  Shared chunkwise machinery.
#
#  Every chunkwise core is the same three-stage pipeline and differs ONLY in
#  how it factorizes the decay ratios entering the score matrices T and A_qk
#  (strategy notes in the module docstring):
#
#    1. preamble          _chunk_inputs      validate C, reshape [L,d]->[N,C,d]
#    2. chunk-local       per-core           decay factorization -> T, Aqk,
#       precompute                           Ebar, Z, Qg, Ktail, gamma_C
#       + WY solve        _wy_solve          Y, U from ONE stacked-RHS solve
#    3. cross-chunk scan  _cross_chunk_scan  the ONLY sequential part
# --------------------------------------------------------------------------- #
def _chunk_inputs(q, k, v, g, b, w, S0, chunk_size):
    """Shared preamble: validate the chunking and reshape every [L, d] input
    to [N, C, d] in D_TYPE — one leading axis per chunk, so every chunk-local
    quantity is computed for all N chunks at once. The within-chunk cumsum
    restarting at each boundary is what realizes both γ_0 = 1 (Eq. 18/30) and
    the normalized init Ŝ_0 = S_[n] (Eq. 31).

    Returns ((q, k, v, g, b, w) chunked, S0 in D_TYPE)."""
    L = k.shape[0]
    C = chunk_size
    if C <= 0 or L % C:
        raise ValueError(
            f"chunk_size={C} must be a positive divisor of the sequence "
            f"length L={L}")
    N = L // C

    def to_chunks(x):
        return x.reshape(N, C, x.shape[-1]).astype(D_TYPE)

    return tuple(to_chunks(x) for x in (q, k, v, g, b, w)), S0.astype(D_TYPE)


def _wy_solve(T: jax.Array, Ebar: jax.Array, Z: jax.Array):
    """WY auxiliaries  Y = (I + T)^{-1} Ē  and  U = (I + T)^{-1} Z  from ONE
    forward substitution over the stacked right-hand side [Ē | Z]
    (Eqs. 21/34 + 22/34). Used by every chunkwise core except the faithful
    one (which keeps the paper's explicit inverse verbatim).

    HOW Eq. 21/34 and Eq. 22/34 WORK. The chunk's residuals obey
    ρ_r = (z_r − ē_rᵀ S0) − Σ_{s<r} T_rs ρ_s (Eq. 38): before edit r can be
    applied, it must subtract its overlap T_rs with every EARLIER edit s it
    partially erases. Stacking the C rows turns that chain of corrections
    into one linear system, (I + T) R = Z − Ē S0 (Eq. 39). T is strictly
    lower triangular (causality), so I + T is unit lower-triangular and
    invertible by construction — Eq. 21/34 defines A = (I + T)^{-1}, and
    Eq. 22/34 pushes A onto the two S0-independent right-hand sides:
    Y = A Ē (erase side), U = A Z (write side), so that later
    R = U − Y S0 assembles the residuals for ANY chunk-entry state.

    HOW THE ROW RECURRENCES OF App. A.4 WORK. Expanding (I + T) Y = Ē and
    (I + T) U = Z row by row gives
        y_rᵀ = ē_rᵀ − Σ_{s<r} (ē_rᵀ k̄_s) y_sᵀ,      Eq. 45
        u_rᵀ = z_rᵀ − Σ_{s<r} (ē_rᵀ k̄_s) u_sᵀ,      Eq. 46
    i.e. row r starts from its own factor (ē_r or z_r) and subtracts each
    earlier, ALREADY-CORRECTED row s weighted by the overlap T_rs = ē_rᵀ k̄_s.
    Computing row 1, then row 2 from row 1, then row 3 from rows 1-2, ... is
    exactly forward substitution — solve_triangular below runs that C-step
    chain in one call. Both recurrences share the SAME coefficients T and
    differ only in the starting vectors, which is why one solve over the
    stacked RHS [Ē | Z] yields both auxiliaries at once (App. A.4: "the same
    WY inverse can be shared by the erase-side and write-side computations").

    The explicit inverse A is never materialized: that would cost a C-RHS
    solve plus two matmuls and is numerically worse than solving directly.
    Y and U are independent of S0 — that is what lets all chunks precompute
    them in parallel before the sequential scan.

    T: [N, C, C]; Ebar: [N, C, dk]; Z: [N, C, dv] -> (Y [N,C,dk], U [N,C,dv]).
    """
    dk = Ebar.shape[-1]
    eye = jnp.eye(T.shape[-1], dtype=T.dtype)
    YU = jax.scipy.linalg.solve_triangular(
        eye + T, jnp.concatenate([Ebar, Z], axis=-1),
        lower=True, unit_diagonal=True,
    )
    return YU[..., :dk], YU[..., dk:]


def _cross_chunk_scan(S0, Y, U, Aqk, Qg, Ktail, gamma_C, delta=None):
    """Cross-chunk recurrence — the ONLY sequential part, identical in every
    chunkwise core. S is the raw chunk-entry state S_[n] (NOT decay-
    normalized); each of the N steps is three small matmuls over the
    per-chunk slices Y_n [C, dk], U_n [C, dv], Aqk_n [C, C], Qg_n [C, dk],
    Ktail_n [C, dk], gamma_C_n [dk].

    `delta` is the centered core's per-chunk re-attachment factor
    exp(G_C/2) ≤ 1: when given, S0 is pre-scaled per chunk as diag(delta_n) S
    so that Y_n S_c == (A Ē) S0 and Qg_n S_c == Q_γ S0 hold exactly. The
    other cores carry the true absolute γ inside Ē and Q_γ and pass None.

    Returns (O flattened back to [L, dv], S_final [dk, dv])."""
    xs = (Y, U, Aqk, Qg, Ktail, gamma_C)
    if delta is not None:
        xs = xs + (delta,)

    def chunk_step(S, inp):
        if delta is None:
            Y_n, U_n, Aqk_n, Qg_n, Ktail_n, gamma_C_n = inp
            S_c = S
        else:
            Y_n, U_n, Aqk_n, Qg_n, Ktail_n, gamma_C_n, delta_n = inp
            S_c = delta_n[:, None] * S  # diag(exp(c)) S0 — re-attach the shift

        # Eq. 35:  R = U − Y S0.                                    [C, dv]
        # Row r is the residual ρ_r = z_r − ē_rᵀ Ŝ_{r−1} (Eq. 37): what is
        # left to write at step r after subtracting what the decayed,
        # already-edited memory returns along the erase direction. U covers
        # the intra-chunk part; −Y S0 corrects for inherited content. R is
        # exactly the chunk's set of rank-one updates:
        # Ŝ_r = S0 + Σ_{s≤r} k̄_s ρ_sᵀ (Eq. 36).
        R = U_n - Y_n @ S_c

        # Eq. 24/44:  O = Q_γ S0 + A_qk R.                          [C, dv]
        # Two read paths: history (query reads the chunk-entry state through
        # its decay γ_r) + intra-chunk (causal scores against this chunk's
        # residual writes). This is o_r = S_rᵀ q_r unrolled through Eq. 36.
        o = Qg_n @ S_c + Aqk_n @ R

        # Eq. 23/40:  S_[n+1] = Diag(γ_C) S0 + K_tailᵀ R.          [dk, dv]
        # Hand-off to the next chunk: old state after a full chunk of decay
        # (erasures folded into R), plus every residual write re-keyed by
        # its tail-decayed key. gamma_C broadcasts over key-channel rows.
        S_new = gamma_C_n[:, None] * S + Ktail_n.T @ R

        return S_new, o

    # scan carries S across chunks; stacked outputs o: [N, C, dv].
    S_final, o = lax.scan(chunk_step, S0, xs)
    return o.reshape(-1, o.shape[-1]), S_final


# --------------------------------------------------------------------------- #
#  The chunkwise cores. Each keeps only its own decay factorization; the
#  preamble, WY solve, and cross-chunk scan live in the helpers above.
# --------------------------------------------------------------------------- #
def _chunkwise_single_faithful(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    b: jax.Array,
    w: jax.Array,
    S0: jax.Array,
    chunk_size: int,
) -> tuple[jax.Array, jax.Array]:
    """Chunkwise WY forward for ONE (batch, head) pair, literal paper form.

    Implements Eqs. 18-25 (main text) = Eqs. 30-44 (App. A) with the exact
    factors printed in the paper, including the explicit inverse
    A = (I + T)^{-1}. Kept as a readable reference; it materializes
    exp(-G), so it overflows fp32 once the within-chunk cumulative
    log-decay G goes below ≈ -88. _chunkwise_single_stacked_RHS_solve is
    this core with only the solver improved; _chunkwise_single_centered is
    the production default.

    Args:
      q, k, g, b: [L, dk]   query / key / log-decay (g ≤ 0) / erase gate
      v, w:       [L, dv]   value / write gate
      S0:         [dk, dv]  state entering the sequence
      chunk_size: C; L must be divisible by C.
    Returns:
      (O: [L, dv], S_final: [dk, dv])
    """
    (q, k, v, g, b, w), S0 = _chunk_inputs(q, k, v, g, b, w, S0, chunk_size)
    eye = jnp.eye(chunk_size, dtype=D_TYPE)  # [C, C]

    # ---- Chunk-local precompute (parallel over the N chunks) -------------

    # Eq. 18/30:  G_r = Σ_{i≤r} g_i (inclusive, within chunk).   [N, C, dk]
    # g ≤ 0 is the per-token log-decay, so γ_r = exp(G_r) = Π_{i≤r} α_i is
    # the total shrinkage a key channel accumulated since chunk start. A
    # write at step s read at step r survives with γ_r/γ_s = exp(G_r − G_s):
    # summing logs turns the running product of gates into a cumsum.
    G = jnp.cumsum(g, axis=1)

    # Eq. 18/30:  γ_r = exp(G_r)                                  [N, C, dk]
    gamma = jnp.exp(G)

    # Total chunk decay γ_C (last row): how much of the chunk-entry state
    # survives to the end of the chunk (the carry in Eq. 23/40).   [N, dk]
    gamma_C = gamma[:, -1]

    # --- Decay normalization: S_r = Diag(γ_r) Ŝ_r (Eq. 31) removes Diag(α)
    # from the recurrence and moves it into the rank-one factors. The ratio
    # γ_r/γ_s is split: γ^{-1} goes on the write side (K̄), γ on the read
    # side (Ē).
    # Eq. 19/32/33:  K̄ = γ^{-1} ⊙ K                              [N, C, dk]
    Kbar = k * jnp.exp(-G)

    # Eq. 20/33:  Ē = γ ⊙ (B ⊙ K), with e_r = b_r ⊙ k_r (Eq. 8). [N, C, dk]
    # Row r pairs with a K̄ row s<r as ē_rᵀ k̄_s = e_rᵀ Diag(γ_r/γ_s) k_s:
    # how much of the association written at s, decayed until r, lies along
    # the gated erase direction e_r.
    Ebar = gamma * (b * k)

    # Eq. 8, 20/33:  Z = W ⊙ V — gated write targets.             [N, C, dv]
    # No γ here: values live on the dv axis, decay acts on key channels.
    Z = w * v

    # Eq. 24/43:  Q_γ, row r = γ_r ⊙ q_r.                         [N, C, dk]
    # The query reads the chunk-entry state decayed down to its own step r.
    Qg = gamma * q

    # --- WY triangular solve (the parallelization) ---
    # Eq. 21/34:  T = tril(Ē K̄ᵀ, −1), T_rs = ē_rᵀ k̄_s (s < r).   [N, C, C]
    # T_rs measures how strongly token r's erase overlaps token s's decayed
    # write. Strictly lower triangular = causality (r only erases the past).
    T = jnp.tril(Ebar @ Kbar.swapaxes(-1, -2), k=-1)

    # Eq. 21/34:  A = (I + T)^{-1}.                                [N, C, C]
    # Residuals obey ρ_r = (z_r − ē_rᵀ S0) − Σ_{s<r} T_rs ρ_s (Eq. 38): each
    # edit first accounts for every earlier edit it partially erases.
    # Stacked: (I + T) R = Z − Ē S0. I + T is unit lower-triangular, so one
    # batched forward substitution replaces the C-step sequential recurrence.
    # The EXPLICIT inverse is the paper's literal form — every other core
    # replaces it with _wy_solve's stacked-RHS forward substitution.
    A = jax.scipy.linalg.solve_triangular(
        eye + T, jnp.broadcast_to(eye, T.shape), lower=True, unit_diagonal=True
    )

    # Eq. 22/34:  Y = A Ē — erase-side auxiliary.                 [N, C, dk]
    Y = A @ Ebar

    # Eq. 22/34:  U = A Z — write-side auxiliary.                 [N, C, dv]
    # Y and U solve the SAME triangular system with different right-hand
    # sides (row recurrences Eqs. 45/46), sharing the one inverse A. Both
    # are independent of S0 — that is what lets all chunks precompute them
    # in parallel before the sequential scan.
    U = A @ Z

    # Eq. 25/43:  (A_qk)_rs = 1_{r≥s} q_rᵀ Diag(γ_r/γ_s) k_s.      [N, C, C]
    # Decay-aware causal attention scores: query r attends to the write at
    # s ≤ r with the key contracted channel-wise by the decay accumulated
    # between s and r. tril INCLUDES the diagonal (a token reads its own
    # write, ratio = 1).
    Aqk = jnp.tril(Qg @ Kbar.swapaxes(-1, -2))

    # Eq. 23/41:  (K_tail)_r = (γ_C/γ_r) ⊙ k_r.                   [N, C, dk]
    # The write at step r keeps decaying for the rest of the chunk, so it
    # enters the end-of-chunk state with the leftover factor γ_C/γ_r.
    # Literal paper form: under strong decay both γ's underflow and this
    # ratio becomes 0/0 = NaN — the other cores form it in log-space as
    # exp(G_C − G_r) instead.
    Ktail = k * (gamma_C[:, None, :] / gamma)

    return _cross_chunk_scan(S0, Y, U, Aqk, Qg, Ktail, gamma_C)


def _chunkwise_single_stacked_RHS_solve(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    b: jax.Array,
    w: jax.Array,
    S0: jax.Array,
    chunk_size: int,
) -> tuple[jax.Array, jax.Array]:
    """Chunkwise WY forward for ONE (batch, head) pair: literal paper
    factors + stacked-RHS solver.

    Identical to _chunkwise_single_faithful in every decay factor — it
    keeps the paper's literal K̄ = exp(−G) ⊙ K and the γ_C/γ ratio for
    K_tail, and therefore shares the same fp32 limits (overflow once the
    within-chunk cumulative log-decay |G_C| > ≈ 88) — but computes the WY
    auxiliaries with ONE triangular solve over the stacked right-hand side
    (see _wy_solve) instead of materializing the explicit inverse.

    This core isolates the SOLVER optimization from the NUMERICAL-RANGE
    ones: diffing it against faithful shows exactly the stacked-RHS change,
    and the centered/pairwise/subchunking cores all build on it.

    Args / returns: identical to _chunkwise_single_faithful.
    """
    (q, k, v, g, b, w), S0 = _chunk_inputs(q, k, v, g, b, w, S0, chunk_size)

    # Literal paper factors, identical to _chunkwise_single_faithful — see
    # its comments for the derivation of each quantity.
    G = jnp.cumsum(g, axis=1)                    # Eq. 18/30       [N, C, dk]
    gamma = jnp.exp(G)                           # Eq. 18/30       [N, C, dk]
    gamma_C = gamma[:, -1]                       # total chunk decay  [N, dk]
    Kbar = k * jnp.exp(-G)                       # Eq. 19/33 (overflow source)
    Ebar = gamma * (b * k)                       # Eq. 20/33       [N, C, dk]
    Z = w * v                                    # Eq. 8, 20/33    [N, C, dv]
    Qg = gamma * q                               # Eq. 24/43       [N, C, dk]
    T = jnp.tril(Ebar @ Kbar.swapaxes(-1, -2), k=-1)  # Eq. 21/34  [N, C, C]

    # Eqs. 21/34 + 22/34 via ONE stacked-RHS forward substitution — the
    # solver optimization this core exists to isolate (see _wy_solve).
    Y, U = _wy_solve(T, Ebar, Z)

    Aqk = jnp.tril(Qg @ Kbar.swapaxes(-1, -2))   # Eq. 25/43       [N, C, C]
    Ktail = k * (gamma_C[:, None, :] / gamma)    # Eq. 23/41 (NaN when γ underflows)

    return _cross_chunk_scan(S0, Y, U, Aqk, Qg, Ktail, gamma_C)


def _chunkwise_single_centered(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    b: jax.Array,
    w: jax.Array,
    S0: jax.Array,
    chunk_size: int,
) -> tuple[jax.Array, jax.Array]:
    """Chunkwise WY forward for ONE (batch, head) pair, exponent-centered.

    Same algebra as _chunkwise_single_faithful (Eqs. 18-25 / 30-44) but
    numerically safe: the faithful form materializes exp(-G) ∈ [1, e^{|G_C|}],
    which overflows fp32 for |G_C| > ~88 per chunk. Every product consumed
    downstream only involves the BOUNDED ratios exp(G_r − G_s), s ≤ r, and
    exp(G_C − G_r), so all within-chunk exponents are shifted by c = G_C/2
    per channel:
      * the shift cancels exactly wherever two centered factors meet
        (T and A_qk); K_tail sidesteps it entirely, being formed directly
        in log-space as exp(G_C − G);
      * it survives only where an absolute γ meets the state (Y S0, Q_γ S0),
        where it is re-attached by pre-scaling S0 with diag(exp(c)),
        exp(c) ≤ 1 — always safe.
    The algebra is exactly the paper's — the shift cancels identically, only
    the floating-point evaluation order changes — and the safe per-chunk
    decay range doubles to |G_C| ≈ 176; beyond that, reduce chunk_size or
    switch to the subchunking/pairwise cores.

    Args / returns: identical to _chunkwise_single_faithful.
    """
    (q, k, v, g, b, w), S0 = _chunk_inputs(q, k, v, g, b, w, S0, chunk_size)

    # Eq. 18/30:  G_r = Σ_{i≤r} g_i, within-chunk cumulative log-decay.
    # [N, C, dk]; see the faithful variant for the γ_r/γ_s reading.
    G = jnp.cumsum(g, axis=1)

    # Total chunk log-decay G_C (last row) and its exp, the carry
    # coefficient of Eq. 23/40. γ_C = exp(G_C) ≤ 1 never overflows. [N, dk]
    G_C = G[:, -1]
    gamma_C = jnp.exp(G_C)

    # Exponent centering: Gc = G − c with c = G_C/2 per channel.  [N, C, dk]
    # G spans [G_C, 0] within a chunk; Gc spans ±|G_C|/2, halving the
    # largest exponent ever materialized.
    Gc = G - 0.5 * G_C[:, None, :]

    # exp(c) ≤ 1: per-chunk state pre-scale that re-attaches the shift
    # against S0 inside the cross-chunk scan.                        [N, dk]
    delta = jnp.exp(0.5 * G_C)

    # Eq. 19/32/33 centered:  K̄ = exp(c−G) ⊙ K (paper: γ^{-1} ⊙ K).
    # [N, C, dk]; max exponent |G_C|/2 instead of |G_C|.
    Kbar = k * jnp.exp(-Gc)

    # Eq. 20/33 centered:  Ē = exp(G−c) ⊙ (B ⊙ K), e_r = b_r ⊙ k_r (Eq. 8).
    # [N, C, dk]. Pairing rows restores the true ratio:
    # ē_rᵀ k̄_s = e_rᵀ Diag(exp(G_r − G_s)) k_s — c cancels.
    Ebar = jnp.exp(Gc) * (b * k)

    # Eq. 8, 20/33:  Z = W ⊙ V — gated write targets (no γ: decay lives on
    # the key axis).                                              [N, C, dv]
    Z = w * v

    # Eq. 24/43 centered:  Q_γ row r = exp(G_r − c) ⊙ q_r.        [N, C, dk]
    # The missing exp(c) is restored by pairing with diag(exp(c)) S0.
    Qg = jnp.exp(Gc) * q

    # Eq. 21/34:  T = tril(Ē K̄ᵀ, −1), T_rs = ē_rᵀ k̄_s (s < r).   [N, C, C]
    # Overlap of edit r with the decayed write s; centering cancels here.
    T = jnp.tril(Ebar @ Kbar.swapaxes(-1, -2), k=-1)

    # Eqs. 21/34 + 22/34: one stacked-RHS forward substitution (_wy_solve).
    Y, U = _wy_solve(T, Ebar, Z)

    # Eq. 25/43:  (A_qk)_rs = 1_{r≥s} q_rᵀ Diag(γ_r/γ_s) k_s.      [N, C, C]
    # Decay-aware causal scores; exp(Gc_r)·exp(−Gc_s) = exp(G_r − G_s), so
    # the centering cancels. tril includes the diagonal (self-read, ratio 1).
    Aqk = jnp.tril(Qg @ Kbar.swapaxes(-1, -2))

    # Eq. 23/41:  (K_tail)_r = (γ_C/γ_r) ⊙ k_r, formed in log-space as
    # exp(G_C − G_r): under strong decay the faithful ratio divides two
    # underflowed denormals (0/0 → NaN).                          [N, C, dk]
    Ktail = k * jnp.exp(G_C[:, None, :] - G)

    # delta re-attaches the centering shift against S0 (see _cross_chunk_scan).
    return _cross_chunk_scan(S0, Y, U, Aqk, Qg, Ktail, gamma_C, delta=delta)


def _chunkwise_single_pairwise(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    b: jax.Array,
    w: jax.Array,
    S0: jax.Array,
    chunk_size: int,
) -> tuple[jax.Array, jax.Array]:
    """Chunkwise WY forward for ONE (batch, head) pair, pairwise log-space.

    Overflow-proof variant of the same algebra (Eqs. 18-25 / 30-44). The
    only overflow source in the chunkwise form is K̄ = exp(−G) ⊙ K, and K̄
    is only ever consumed inside the pairwise products T and A_qk. Here
    those entries are formed directly from the log-space DIFFERENCE,

        exp(G_r − G_s)  per (r, s, key-channel) triple,  s ≤ r,

    which is ≤ 0 in the exponent (G is non-increasing in r), so every
    exponential lies in (0, 1] — no overflow at ANY decay strength, and no
    centering/delta bookkeeping. The remaining γ-carrying factors are
    already safe: Ē = γ⊙(b⊙k) and Q_γ = γ⊙q have exponents G ≤ 0, and
    K_tail uses exp(G_C − G_r) ≤ 1.

    The price: T and A_qk are no longer [C,dk] @ [dk,C] matmuls but einsums
    over a [N, C, C, dk] decay tensor — ×C more memory on the score path
    and no tensor-core mapping. Use this core only when the decay is too
    strong for _chunkwise_single_centered (per-chunk |G_C| > ~176) and
    shrinking chunk_size is not an option; _chunkwise_single_subchunking
    solves the same problem with a better memory/compute profile.

    Args / returns: identical to _chunkwise_single_faithful.
    """
    (q, k, v, g, b, w), S0 = _chunk_inputs(q, k, v, g, b, w, S0, chunk_size)
    C = chunk_size

    # Eq. 18/30:  G_r = Σ_{i≤r} g_i, within-chunk cumulative log-decay.
    # [N, C, dk]; see the faithful variant for the γ_r/γ_s reading.
    G = jnp.cumsum(g, axis=1)

    # Total chunk log-decay G_C (last row) and its exp, the carry
    # coefficient of Eq. 23/40. γ_C = exp(G_C) ≤ 1 never overflows. [N, dk]
    G_C = G[:, -1]
    gamma_C = jnp.exp(G_C)

    # Eq. 8:  e = b ⊙ k — erase directions, shared by T and Ē.   [N, C, dk]
    e = b * k

    # Pairwise log-decay differences G_r − G_s for every (r, s) pair and key
    # channel — the exponent of the Diag(γ_r/γ_s) factor in Eqs. 21/25.
    # [N, C, C, dk] — the ×C memory cost of this variant lives here.
    Gdiff = G[:, :, None, :] - G[:, None, :, :]

    # The mask must be applied BEFORE the exp: anti-causal pairs r < s have
    # G_r − G_s ≥ 0, whose exp can overflow to inf, and a post-hoc tril
    # would then compute inf · 0 = NaN. Masked entries are sent to −inf so
    # exp gives exactly 0.
    # Only ONE [N, C, C, dk] decay tensor is materialized — the inclusive
    # (s ≤ r) one. T's strict (s < r) counterpart differs from it only on
    # the diagonal, where D_incl = exp(0) = 1 (finite, no inf·0 hazard), so
    # T is built with D_incl and the s = r scores are discarded afterwards
    # by a strict tril on the small [N, C, C] result — saving a second
    # full-tensor exp/where pass and its memory.
    causal = jnp.tril(jnp.ones((C, C), dtype=bool))        # s ≤ r
    neg_inf = jnp.array(-jnp.inf, dtype=D_TYPE)
    D_incl = jnp.exp(jnp.where(causal[None, :, :, None], Gdiff, neg_inf))

    # Eq. 21/34:  T_rs = ē_rᵀ k̄_s = Σ_c e_rc k_sc exp(G_rc − G_sc), s < r.
    # Same entries as tril(Ē K̄ᵀ, −1), but the decay enters as a bounded
    # per-triple factor instead of two unbounded row/column scalings — an
    # einsum with a [C, C, dk] operand, not a matmul.            [N, C, C]
    T = jnp.tril(jnp.einsum('nrc,nsc,nrsc->nrs', e, k, D_incl), k=-1)

    # Eq. 25/43:  (A_qk)_rs = q_rᵀ Diag(γ_r/γ_s) k_s, s ≤ r — decay-aware
    # causal attention scores, same pairwise construction (diagonal
    # included: self-read, ratio = 1).                            [N, C, C]
    Aqk = jnp.einsum('nrc,nsc,nrsc->nrs', q, k, D_incl)

    # Absolute-γ factors, exponents G ≤ 0 so exp ∈ (0,1] — always safe;
    # no centering shift and no delta pre-scale are needed in this variant.
    # Eq. 18/30:  γ = exp(G), computed once and shared by Ē and Q_γ.
    gamma = jnp.exp(G)  # [N, C, dk]

    # Eq. 20/33:  Ē = γ ⊙ (B ⊙ K)                                [N, C, dk]
    Ebar = gamma * e

    # Eq. 24/43:  Q_γ, row r = γ_r ⊙ q_r                          [N, C, dk]
    Qg = gamma * q

    # Eq. 8, 20/33:  Z = W ⊙ V — gated write targets.             [N, C, dv]
    Z = w * v

    # Eqs. 21/34 + 22/34: one stacked-RHS forward substitution (_wy_solve).
    Y, U = _wy_solve(T, Ebar, Z)

    # Eq. 23/41:  (K_tail)_r = (γ_C/γ_r) ⊙ k_r, formed in log-space as
    # exp(G_C − G_r) ≤ 1.                                        [N, C, dk]
    Ktail = k * jnp.exp(G_C[:, None, :] - G)

    # Ē and Q_γ carry the true absolute γ here, so no delta pre-scale.
    return _cross_chunk_scan(S0, Y, U, Aqk, Qg, Ktail, gamma_C)


def _chunkwise_single_subchunking(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    b: jax.Array,
    w: jax.Array,
    S0: jax.Array,
    chunk_size: int,
    sub_chunk_size: int,
) -> tuple[jax.Array, jax.Array]:
    """Chunkwise WY forward for ONE (batch, head) pair, secondary chunking.

    GLA-style middle ground between the centered and pairwise cores. The
    triangular solve, WY auxiliaries, and cross-chunk scan are unchanged at
    full chunk size C; only the construction of the decay-aware score
    matrices T (Eq. 21/34) and A_qk (Eq. 25/43) differs. Each chunk is split
    into M = C/c sub-blocks of size c, and the entry (r, s) is built by
    where the pair falls:

      * OFF-DIAGONAL block pair (row r in sub-block i, column s in an
        earlier sub-block j < i): factor the decay ratio through the
        row-block's entry boundary B_i = G at the last position before
        sub-block i,
            exp(G_r − G_s) = exp(G_r − B_i) · exp(B_i − G_s),
        and since s ≤ (boundary) ≤ r and G is non-increasing, BOTH
        exponents are ≤ 0 — each factor lies in (0, 1], so these blocks are
        ordinary batched matmuls with no overflow at any decay strength.
      * DIAGONAL block (r, s in the same sub-block): the boundary trick
        cannot help (no boundary between them), so use the pairwise
        log-space form exp(G_r − G_s) per (r, s, channel) triple — but only
        over c×c blocks, a [N, M, c, c, dk] intermediate (×c memory)
        instead of the pairwise core's [N, C, C, dk] (×C).

    Result: NO decay-range limit (like the pairwise core), matmul-dominated
    compute (like the centered core), memory overhead ×(C/c) on the
    rescaled-key tensor + ×c on the diagonal blocks. This is the
    block-decomposition production Triton/CUDA kernels use.

    Args: as _chunkwise_single_faithful, plus
      sub_chunk_size: c; must divide chunk_size.
    Returns:
      (O: [L, dv], S_final: [dk, dv])
    """
    C = chunk_size
    c = sub_chunk_size
    if c <= 0 or C % c:
        raise ValueError(
            f"sub_chunk_size={c} must be a positive divisor of chunk_size={C}")
    (q, k, v, g, b, w), S0 = _chunk_inputs(q, k, v, g, b, w, S0, chunk_size)
    N, _, dk = k.shape
    M = C // c   # sub-blocks per chunk

    e = b * k                       # Eq. 8: erase directions   [N, C, dk]

    # Eq. 18/30:  G_r = Σ_{i≤r} g_i, within-chunk cumulative log-decay.
    G = jnp.cumsum(g, axis=1)       # [N, C, dk]

    # Total chunk log-decay and carry coefficient of Eq. 23/40.     [N, dk]
    G_C = G[:, -1]
    gamma_C = jnp.exp(G_C)

    # ---- Secondary-chunking decomposition of the decay ratios -------------
    # Sub-block view of G and the entry boundary B_i of each sub-block
    # (cumulative log-decay at the last position BEFORE the block; B_0 = 0,
    # matching γ_0 = 1 of Eq. 18/30 at the second level).
    G4 = G.reshape(N, M, c, dk)                                  # [N, M, c, dk]
    B = jnp.concatenate(
        [jnp.zeros((N, 1, dk), dtype=D_TYPE), G4[:, :-1, -1]], axis=1
    )                                                            # [N, M, dk]

    # Block-local log-decay Grel_r = G_r − B_i ≤ 0 (r is inside block i, at
    # or after its boundary), so exp(Grel) ∈ (0, 1] — always safe.
    Grel = G4 - B[:, :, None, :]                                 # [N, M, c, dk]

    # Row-side factors carrying exp(G_r − B_i): the "Ē/Q_γ of the sub-block".
    Erow = e.reshape(N, M, c, dk) * jnp.exp(Grel)                # [N, M, c, dk]
    Qrow = q.reshape(N, M, c, dk) * jnp.exp(Grel)                # [N, M, c, dk]

    # Column-side keys rescaled per ROW-block: exp(B_i − G_s) ≤ 1 for s in
    # any earlier sub-block. Masked to −inf BEFORE the exp — for s at or
    # after block i the exponent is ≥ 0 and could overflow; exp(−inf) = 0
    # also implements the block-level causal mask. This [N, M, C, dk] tensor
    # is the ×(C/c) memory cost of the variant.
    neg_inf = jnp.array(-jnp.inf, dtype=D_TYPE)
    col_blk = jnp.arange(C) // c                                 # [C] block of s
    past = col_blk[None, :] < jnp.arange(M)[:, None]             # [M, C] j < i
    expo = jnp.where(past[None, :, :, None],
                     B[:, :, None, :] - G[:, None, :, :], neg_inf)
    Ksc = k[:, None, :, :] * jnp.exp(expo)                       # [N, M, C, dk]

    # OFF-DIAGONAL blocks of Eq. 21/34 and Eq. 25/43: for each row-block i,
    # T_rs = (e_r exp(G_r − B_i))ᵀ (k_s exp(B_i − G_s)) — a batched matmul
    # [M, c, dk] @ [M, dk, C]; columns in blocks ≥ i are exactly 0.
    T_off = jnp.einsum('nmrd,nmsd->nmrs', Erow, Ksc).reshape(N, C, C)
    Aqk_off = jnp.einsum('nmrd,nmsd->nmrs', Qrow, Ksc).reshape(N, C, C)

    # DIAGONAL c×c blocks: pairwise log-space differences within each
    # sub-block (B_i cancels: Grel_r − Grel_s = G_r − G_s). Exponents ≤ 0
    # under the causal mask, applied before the exp as in the pairwise
    # core. Only the inclusive (s ≤ r) decay tensor is materialized: its
    # strict counterpart differs only on the diagonal, where
    # D_incl = exp(0) = 1 (finite, no inf·0 hazard), so T_diag is built
    # with D_incl and the s = r scores are discarded by a strict tril on
    # the small [N, M, c, c] result.                     Gd: [N, M, c, c, dk]
    Gd = Grel[:, :, :, None, :] - Grel[:, :, None, :, :]
    causal = jnp.tril(
        jnp.ones((c, c), dtype=bool)
    )       # s <= r
    D_incl = jnp.exp(
        jnp.where(causal[None, None, :, :, None], Gd, neg_inf)
    )
    kg = k.reshape(N, M, c, dk)
    T_diag = jnp.tril(
        jnp.einsum(
            'nmrc,nmsc,nmrsc->nmrs',
            e.reshape(N, M, c, dk), kg, D_incl),
        k=-1)                                                   # [N, M, c, c]
    Aqk_diag = jnp.einsum('nmrc,nmsc,nmrsc->nmrs',
                          q.reshape(N, M, c, dk), kg, D_incl)   # [N, M, c, c]

    # Scatter the diagonal blocks into the [C, C] score matrices (the
    # off-diagonal parts are 0 there, so add == set).
    rr = jnp.arange(M)[:, None, None] * c + jnp.arange(c)[None, :, None]
    ss = jnp.arange(M)[:, None, None] * c + jnp.arange(c)[None, None, :]
    T = T_off.at[:, rr, ss].add(T_diag)      # Eq. 21/34          [N, C, C]
    Aqk = Aqk_off.at[:, rr, ss].add(Aqk_diag)  # Eq. 25/43        [N, C, C]

    # ---- From here on identical to the pairwise core ----------------------
    # Absolute-γ factors, exponents G ≤ 0 so exp ∈ (0,1] — always safe.
    gamma = jnp.exp(G)      # Eq. 18/30:  γ = exp(G), shared     [N, C, dk]
    Ebar = gamma * e        # Eq. 20/33:  Ē = γ ⊙ (B ⊙ K)        [N, C, dk]
    Qg = gamma * q          # Eq. 24/43:  Q_γ = γ ⊙ Q            [N, C, dk]
    Z = w * v               # Eq. 8, 20/33:  Z = W ⊙ V           [N, C, dv]

    # Eqs. 21/34 + 22/34, solved at FULL chunk size C (the sub-chunking only
    # changed how T's entries were produced): one stacked-RHS forward
    # substitution (_wy_solve).
    Y, U = _wy_solve(T, Ebar, Z)

    # Eq. 23/41:  (K_tail)_r = exp(G_C − G_r) ⊙ k_r ≤ k_r.       [N, C, dk]
    Ktail = k * jnp.exp(G_C[:, None, :] - G)

    return _cross_chunk_scan(S0, Y, U, Aqk, Qg, Ktail, gamma_C)


def _batchify(fn):
    """Lift a per-head function to batched [B, H, ...] inputs.

    Pure plumbing, no math: vmap over heads (axis 1), then over batch
    (axis 0). All 7 arguments map over their leading axis; S0 simply has no
    L axis.
    """
    over_heads = jax.vmap(fn, in_axes=(0, 0, 0, 0, 0, 0, 0), out_axes=(0, 0))
    return jax.vmap(over_heads, in_axes=(0, 0, 0, 0, 0, 0, 0), out_axes=(0, 0))


# Registry backing the public entry point's core= selector. Keys are the
# accepted `core` names; see chunkwise_gated_delta_rule_2 for guidance on
# choosing one.
_CHUNKWISE_CORES = {
    "faithful": _chunkwise_single_faithful,
    "stacked_rhs": _chunkwise_single_stacked_RHS_solve,
    "centered": _chunkwise_single_centered,
    "pairwise": _chunkwise_single_pairwise,
    "subchunking": _chunkwise_single_subchunking,
}


def chunkwise_gated_delta_rule_2(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    b: jax.Array,
    w: jax.Array,
    S0: jax.Array,
    chunk_size: int = 64,
    core: str = "centered",
    sub_chunk_size: int = 16,
) -> tuple[jax.Array, jax.Array]:
    """Chunkwise-parallel Gated Delta Rule-2 forward (training path).

    Args:
      q, k, g, b: [B, H, L, dk]   v, w: [B, H, L, dv]   S0: [B, H, dk, dv]
      chunk_size: intra-chunk length C (L must be divisible by C).
      core: which chunkwise core computes each head —
        "centered"     (default) exponent-centered matmuls; safe for
                       per-chunk cumulative log-decay |G_C| up to ≈ 176.
        "subchunking"  GLA-style secondary chunking; NO decay-range limit,
                       matmul-dominated; tuned by sub_chunk_size.
        "pairwise"     per-triple log-space decay; NO range limit but ×C
                       memory on the score path — verification / fallback.
        "stacked_rhs"  the paper's literal factors + stacked-RHS solver;
                       overflows past |G_C| ≈ 88 — reference.
        "faithful"     the paper's equations verbatim (explicit inverse);
                       overflows past |G_C| ≈ 88 — reference.
      sub_chunk_size: c for core="subchunking" (must divide chunk_size);
                      ignored by every other core.
    Returns:
      (O: [B, H, L, dv], S_final: [B, H, dk, dv])
    """
    if core not in _CHUNKWISE_CORES:
        raise ValueError(
            f"core={core!r} is not one of {sorted(_CHUNKWISE_CORES)}")

    kwargs = {"chunk_size": chunk_size}
    if core == "subchunking":
        kwargs["sub_chunk_size"] = sub_chunk_size
    fun = partial(_CHUNKWISE_CORES[core], **kwargs)
    return _batchify(fun)(q, k, v, g, b, w, S0)


def recurrent_gated_delta_rule_2(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    b: jax.Array,
    w: jax.Array,
    S0: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Token-by-token Gated Delta Rule-2 forward (reference / inference path).

    Same I/O contract as chunkwise_gated_delta_rule_2 (without chunk_size):
      q, k, g, b: [B, H, L, dk]   v, w: [B, H, L, dv]   S0: [B, H, dk, dv]
    Returns:
      (O: [B, H, L, dv], S_final: [B, H, dk, dv])
    """
    return _batchify(_recurrent_single)(q, k, v, g, b, w, S0)
