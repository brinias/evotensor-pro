#%%writefile /kaggle/working/symbolics_integration.py
# -*- coding: utf-8 -*-
"""
Symbolics integration — Kaggle/JAX-safe (no CLI, no Python branching in jitted paths).
- Provides:
    * DEFAULT_PROTEIN_CONFIG
    * SymbolicsHelper(...).build_for_subset(seqs, max_len)
    * ensure_symbolic_params(params, num_symbols, rng_key)
    * make_train_step_symbolic(optimizer)
    * make_train_step_symbolic_pmap(optimizer, axis_name="dp")   <-- multi-GPU
    * get_pairs_for_batch(batch_indices, neighbors_local, ...)
- Includes encoder/decoder implementations that match the trainer.
"""

from __future__ import annotations
import re
from typing import List, Dict, Tuple, Optional

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import optax

# -------------------------- Minimal defaults ---------------------------
DEFAULT_PROTEIN_CONFIG = {
    "alphabet": "protein",
    "k_neighbors": 8,
    "symbols": [
        {"id": "P_LOOP_WA",    "regex": r"G....GKT[ST]"},
        {"id": "HExH_METALLO", "regex": r"H..H"},
        {"id": "CYS_KNOT",     "regex": r"C.{2}C.{4,8}C"},
    ],
}

# ---------------------- Dropout + Encode/Decode (mirror trainer) --------------------
def apply_dropout(x, key, rate):
    rate = jnp.asarray(rate, dtype=x.dtype)
    keep_prob = jnp.clip(1.0 - rate, 0.0, 1.0)
    def do_keep(_):
        mask = jax.random.bernoulli(key, p=keep_prob, shape=x.shape)
        return jnp.where(mask, x / jnp.maximum(keep_prob, 1e-6), 0.0)
    return lax.cond(jnp.equal(rate, 0.0), lambda _: x, do_keep, operand=None)

def _encode_step(carry, token_and_pos):
    state, mps, rng_key, rate = carry
    tok, pos = token_and_pos
    W = mps[pos, tok]
    state = state @ W
    key = jax.random.fold_in(rng_key, pos)
    state = apply_dropout(state, key, rate)
    return (state, mps, rng_key, rate), state

@jax.jit
def encode(enc_params, seq_indices, rng_key, dropout_rate):
    L = enc_params['mps'].shape[0]
    bond_dim_mps = enc_params['mps'].shape[-1]
    state0 = jnp.zeros((bond_dim_mps,), dtype=jnp.float32).at[0].set(1.0)
    positions = jnp.arange(L, dtype=jnp.int32)
    (_, _, _, _), states = lax.scan(
        _encode_step,
        (state0, enc_params['mps'], rng_key, dropout_rate),
        (seq_indices, positions)
    )
    # MERA pairwise reduction
    def pair_reduce(_, i):
        l = states[2*i]
        r = states[2*i + 1]
        iso = enc_params['isos'][i]
        hi = jnp.einsum("i,j,ijk->k", l, r, iso)
        return None, hi
    _, hi_list = lax.scan(pair_reduce, None, jnp.arange(L//2))
    tv = hi_list.reshape(-1)
    return tv / (jnp.linalg.norm(tv) + 1e-9)

def _decode_step(carry, tok_and_pos):
    vec, ctx, mps_dec, out_proj, rng_key, rate, first_flag = carry
    tok, pos = tok_and_pos
    W = mps_dec[pos, tok]
    vec = vec @ W
    vec = vec + jnp.where(first_flag, ctx, 0.0)
    key = jax.random.fold_in(rng_key, 1000 + pos)
    vec = apply_dropout(vec, key, rate)
    logits = vec @ out_proj
    first_flag = False
    return (vec, ctx, mps_dec, out_proj, rng_key, rate, first_flag), logits

@jax.jit
def decode_teacher_forcing(params, thought_vector, decoder_input, rng_key, dropout_rate):
    L = params['decoder']['mps'].shape[0]
    bond_dim_mps = params['decoder']['mps'].shape[-1]
    ctx = thought_vector @ params['projection_matrix']
    vec0 = jnp.zeros((bond_dim_mps,), dtype=jnp.float32).at[0].set(1.0)
    positions = jnp.arange(L-1, dtype=jnp.int32)
    mps_dec = params['decoder']['mps'][:-1]
    (_, _, _, _, _, _, _), logits = lax.scan(
        _decode_step,
        (vec0, ctx, mps_dec, params['output_projection'], rng_key, dropout_rate, True),
        (decoder_input, positions)
    )
    return logits  # (L-1, V)

# ---------------------- Symbolic pattern tooling ------------------------
class SymbolicsHelper:
    """
    Lightweight symbolic signals + neighbor graph for a *subset* of sequences.
    - Builds only for the sequences you pass (no giant global arrays).
    - Masks default to ones (no hard forbids) to avoid aggressive constraints.
    """
    def __init__(self, processor, config: Dict):
        self.proc = processor
        self.config = dict(config or {})
        self.alphabet = self.config.get('alphabet', 'protein')
        self.k_neighbors = int(self.config.get('k_neighbors', 8))
        self.symbol_defs = list(self.config.get('symbols', []))
        self.symbol_ids = [d['id'] for d in self.symbol_defs]
        self.compiled = [re.compile(d['regex']) for d in self.symbol_defs]

    @property
    def num_symbols(self) -> int:
        return len(self.symbol_ids)

    def build_for_subset(self, seq_list: List[List[str]], max_len: int):
        """Return (masks_u8, symv_u8, neighbors_local) for the provided seqs."""
        V = int(self.proc.word_count)
        Lm1 = int(max_len - 1)
        S = self.num_symbols
        B = len(seq_list)

        masks = np.ones((B, Lm1, V), dtype=np.uint8)
        symv  = np.zeros((B, S), dtype=np.uint8) if S > 0 else np.zeros((B,1), np.uint8)

        strings = [''.join(s) for s in seq_list]
        # symbol presence
        for i, st in enumerate(strings):
            for s_idx, rgx in enumerate(self.compiled):
                if rgx.search(st) is not None:
                    symv[i, s_idx] = 1

        # neighbor graph from co-membership
        if S > 0 and self.k_neighbors > 0:
            idxs_by_symbol: Dict[int, List[int]] = {s: [] for s in range(S)}
            for i, st in enumerate(strings):
                for s_idx, rgx in enumerate(self.compiled):
                    if rgx.search(st) is not None:
                        idxs_by_symbol[s_idx].append(i)
            neighbor_sets = [set() for _ in range(B)]
            for s, members in idxs_by_symbol.items():
                if len(members) <= 1:
                    continue
                for pos, i in enumerate(members):
                    for off in range(1, min(self.k_neighbors, len(members)-1) + 1):
                        j = members[(pos + off) % len(members)]
                        if j != i: neighbor_sets[i].add(j)
            K = int(self.k_neighbors)
            neighbors_local = np.full((B, K), -1, dtype=np.int32)
            for i, sset in enumerate(neighbor_sets):
                if not sset: continue
                take = list(sset)[:K]
                neighbors_local[i, :len(take)] = np.array(take, dtype=np.int32)
        else:
            neighbors_local = np.full((B, 1), -1, dtype=np.int32)

        return masks, symv, neighbors_local

# ---------------------- Train-time utilities ------------------------

def ensure_symbolic_params(params: Dict, num_symbols: int, rng_key) -> Dict:
    """Ensure params['symbolic_tv_proj'] exists (shape: [S, thought_dim])."""
    if num_symbols <= 0:
        return params
    thought_dim = int(params['projection_matrix'].shape[0])
    need_init = ('symbolic_tv_proj' not in params) or (params['symbolic_tv_proj'].shape != (num_symbols, thought_dim))
    if need_init:
        k = jax.random.fold_in(rng_key, 777)
        params['symbolic_tv_proj'] = jax.random.normal(k, (num_symbols, thought_dim)) * 0.05
    return params

@jax.jit
def _xent_logits_with_int_labels(logits, labels_int, label_smoothing=0.0):
    V = logits.shape[-1]  # make num_classes concrete for JAX
    ls = jnp.asarray(label_smoothing, dtype=logits.dtype)
    onehot = jax.nn.one_hot(labels_int, V)
    smoothed = optax.smooth_labels(onehot, ls)
    return optax.softmax_cross_entropy(logits=logits, labels=smoothed)

@jax.jit
def loss_fn_symbolic(params: Dict,
                     batch_inputs: jnp.ndarray,
                     pad_idx: int,
                     rng_key,
                     dropout_rate: float,
                     label_smoothing: float,
                     allowed_masks_uint8: jnp.ndarray,   # (B, L-1, V) uint8
                     symbol_vecs_uint8: jnp.ndarray,     # (B, S)    uint8
                     pairs_padded: jnp.ndarray,          # (M, 2)    int32
                     pairs_mask: jnp.ndarray,            # (M,)      int32 {0,1}
                     lambda_graph: float = 0.0,
                     alpha: float = 1.0,
                     tau: float = 1.0) -> jnp.ndarray:
    """
    Total loss = CE + soft-constraint penalty + lambda_graph * Laplacian(tv).
      - tv := encode(seq) + alpha * (symv @ symbolic_tv_proj)
      - soft-constraint: encourage probability mass on allowed tokens
        with softening factor tau in [0,1]:
          allowed_soft = 1 - (1 - allowed) * tau
        tau=1 -> strict original mask; tau=0 -> all-ones (no penalty)
    """
    B = batch_inputs.shape[0]

    # Encode
    enc_keys = jax.random.split(rng_key, B)
    dec_keys = jax.random.split(jax.random.fold_in(rng_key, 12345), B)
    tvs = jax.vmap(encode, in_axes=(None,0,0,None))(
        params['encoder'], batch_inputs, enc_keys, dropout_rate
    )  # (B, thought_dim)

    # Add symbolic projection (if present)
    sv = symbol_vecs_uint8.astype(jnp.float32)
    if 'symbolic_tv_proj' in params:
        alpha_f = jnp.asarray(alpha, dtype=tvs.dtype)
        tvs = tvs + alpha_f * (sv @ params['symbolic_tv_proj'])

    # Decode
    dec_in  = batch_inputs[:, :-1]
    dec_tgt = batch_inputs[:, 1:]
    logits  = jax.vmap(decode_teacher_forcing, in_axes=(None,0,0,0,None))(
        params, tvs, dec_in, dec_keys, dropout_rate
    )  # (B, L-1, V)

    # CE
    xent = _xent_logits_with_int_labels(logits, dec_tgt, label_smoothing)  # (B,L-1)
    mask_tok = (dec_tgt != pad_idx)

    # Soft rule-penalty
    probs = jax.nn.softmax(logits, axis=-1)
    allowed = allowed_masks_uint8.astype(probs.dtype)
    tau_f = jnp.clip(jnp.asarray(tau, dtype=probs.dtype), 0.0, 1.0)
    allowed_soft = 1.0 - (1.0 - allowed) * tau_f
    allowed_mass = (probs * allowed_soft).sum(axis=-1)  # (B,L-1)
    rule_pen = 1.0 - allowed_mass

    token_losses = xent + rule_pen
    per_seq_sum = (token_losses * mask_tok).sum(axis=1)
    per_seq_cnt = mask_tok.sum(axis=1)
    token_loss = (per_seq_sum.sum() / jnp.maximum(per_seq_cnt.sum(), 1.0))

    # Graph Laplacian on tvs (pairs within batch)
    i = pairs_padded[:,0]; j = pairs_padded[:,1]
    valid = (pairs_mask > 0).astype(tvs.dtype)
    i = jnp.where(valid > 0, i, 0)
    j = jnp.where(valid > 0, j, 0)
    diffs = tvs[i] - tvs[j]
    gl = ((diffs**2).sum(axis=1) * valid).sum() / jnp.maximum(valid.sum(), 1.0)

    return token_loss + (jnp.asarray(lambda_graph, dtype=tvs.dtype) * gl)

def make_train_step_symbolic(optimizer):
    """Single-device train step."""
    @jax.jit
    def step(params: Dict, opt_state,
             batch_inputs: jnp.ndarray,
             pad_idx: int,
             rng_key,
             dropout_rate: float,
             label_smoothing: float,
             allowed_masks_uint8: jnp.ndarray,
             symbol_vecs_uint8: jnp.ndarray,
             pairs_padded: jnp.ndarray,
             pairs_mask: jnp.ndarray,
             lambda_graph: float = 0.0,
             alpha: float = 1.0,
             tau: float = 1.0):
        def _loss(p):
            return loss_fn_symbolic(
                p, batch_inputs, pad_idx, rng_key, dropout_rate, label_smoothing,
                allowed_masks_uint8, symbol_vecs_uint8, pairs_padded, pairs_mask,
                lambda_graph, alpha, tau
            )
        l, grads = jax.value_and_grad(_loss)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, l
    return step

def make_train_step_symbolic_pmap(optimizer, axis_name: str = "dp"):
    """
    Multi-GPU train step using pmap (data-parallel).
    - params/opt_state are broadcast (in_axes=None)
    - batch and per-batch extras are sharded on leading axis (in_axes=0)
    - grads and loss are averaged with lax.pmean across devices
    - returns unsharded params/opt_state/loss (out_axes=None) for easy checkpointing
    """
    def step_impl(params: Dict, opt_state,
                  batch_inputs: jnp.ndarray,
                  pad_idx: int,
                  rng_key,                       # per-device key
                  dropout_rate: float,
                  label_smoothing: float,
                  allowed_masks_uint8: jnp.ndarray,
                  symbol_vecs_uint8: jnp.ndarray,
                  pairs_padded: jnp.ndarray,
                  pairs_mask: jnp.ndarray,
                  lambda_graph: float = 0.0,
                  alpha: float = 1.0,
                  tau: float = 1.0):
        def _loss(p):
            return loss_fn_symbolic(
                p, batch_inputs, pad_idx, rng_key, dropout_rate, label_smoothing,
                allowed_masks_uint8, symbol_vecs_uint8, pairs_padded, pairs_mask,
                lambda_graph, alpha, tau
            )
        l, grads = jax.value_and_grad(_loss)(params)
        grads = lax.pmean(grads, axis_name=axis_name)
        l = lax.pmean(l, axis_name=axis_name)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, l

    pstep = jax.pmap(
        step_impl,
        axis_name=axis_name,
        in_axes=(None, None, 0, None, 0, None, None, 0, 0, 0, 0, None, None, None),
        out_axes=(None, None, None),
    )
    return pstep

# --------------- batch-edge builder (graph pairs per batch) --------------
def get_pairs_for_batch(batch_indices: np.ndarray,
                        neighbors_local: np.ndarray,
                        edges_per_node: int = 4,
                        max_pairs: int = 2048) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Build a fixed-size (max_pairs,2) set of (i,j) edges within the batch for Laplacian reg.
    neighbors_local is (N_split, K) with local indices for the *split*.
    """
    loc_map = {int(j): b for b, j in enumerate(batch_indices)}
    pairs = []
    for b, j in enumerate(batch_indices):
        cnt = 0
        for nbj in neighbors_local[int(j)]:
            if nbj < 0: continue
            if int(nbj) in loc_map:
                pairs.append([b, loc_map[int(nbj)]])
                cnt += 1
                if cnt >= edges_per_node: break
        if len(pairs) >= max_pairs:
            break
    pairs_padded = np.zeros((max_pairs, 2), dtype=np.int32)
    pairs_mask   = np.zeros((max_pairs,), dtype=np.int32)
    if pairs:
        k = min(len(pairs), max_pairs)
        pairs_padded[:k] = np.array(pairs[:k], dtype=np.int32)
        pairs_mask[:k] = 1
    return jnp.asarray(pairs_padded), jnp.asarray(pairs_mask)
