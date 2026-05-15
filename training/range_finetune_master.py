#%%writefile /kaggle/working/range_finetune_master.py
# -*- coding: utf-8 -*-
# ==============================================================================
#   MERA-MPS range-aware fine-tuning (AMP-ready)
#   - Loads base LM from production_model_bio.pkl
#   - Filters Pfam by length window; optional upsample near target length
#   - Two phases: A) freeze encoder, B) unfreeze all
#   - Grows/shrinks to TARGET_LEN; saves tuned model (production_model_tuned.pkl)
#   - Deterministic-ish, memory-safe (JAX preallocate OFF)
# ==============================================================================

import os, re, time, math, random, pickle, warnings
from typing import List, Tuple, Optional

# ---- JAX memory safety (must be set BEFORE importing jax) ----
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.80")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "default")

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import jax.tree_util as jtu
import optax
warnings.filterwarnings("ignore", category=UserWarning)

# ========================== USER SETTINGS (tuned for AMP) ====================
BASE_MODEL   = os.environ.get("BASE_MODEL", "/kaggle/input/model1/other/default/1/production_model_bio.pkl")
TUNED_MODEL  = os.environ.get("TUNED_MODEL", "/kaggle/working/production_model_tuned.pkl")

PFAM_FILES   = [
    "/kaggle/input/pfaseed/Pfam-A.seed",
    "/kaggle/input/pfafull/Pfam-A.full",   # αν λείπει, απλά προχωράμε
]

# ---- Εύρος για AMPs (συνήθως 24–50 aa). Ρύθμισε αν θες πιο φαρδύ range.
KEEP_WINDOW: Tuple[int,int] = (24, 52)    # inclusive

# Αν None → αυτόματο mid-even. Για AMPs ένα καλό anchor είναι 40.
TARGET_LEN: Optional[int] = 40

# Upsampling γύρω από το anchor length (σταθεροποιεί το tokenizer και το MERA)
NEAR_DELTA        = 2
NEAR_TARGET_MIN   = 1800   # αυξάνει βήματα/χρόνο, αλλά δίνει πιο σταθερό μοντέλο

# Batch sizing
TOKENS_PER_BATCH  = 8192
BATCH_MAX         = 64

# Reproducibility
PY_SEED           = 0

# Phase A (freeze encoder)
EPOCHS_A   = 12
LR_A       = 1.0e-4
DROPOUT_A  = 0.10
SMOOTH_A   = 0.02
PATIENCE_A = 4

# Phase B (unfreeze all)
EPOCHS_B   = 120
LR_B       = 8.0e-5
DROPOUT_B  = 0.10
SMOOTH_B   = 0.02
PATIENCE_B = 8

# Optional capacity bump (None = keep as-is)
BOND_DIM_MPS_TUNED: Optional[int]  = None   # e.g. 48
BOND_DIM_MERA_TUNED: Optional[int] = None   # e.g. 48

# ========================= FORMAT CHECK ======================================
def is_compact(params):
    try:
        return ('encoder' in params and isinstance(params['encoder'], dict)
                and 'mps' in params['encoder'] and 'isos' in params['encoder'])
    except Exception:
        return False

# ========================= DATA HELPERS ======================================
VALID_AA = set('ACDEFGHIKLMNPQRSTVWY')

def _is_seq_line(line: str) -> bool:
    if (not line) or line.startswith('#') or line.startswith('//'):
        return False
    parts = line.split()
    if len(parts) < 2:
        return False
    seq = parts[-1]
    return bool(re.fullmatch(r'[A-Za-z\-\.]+', seq))

def read_pfam_sequences(file_paths: List[str], max_len_cap: int=256) -> List[List[str]]:
    seqs = []
    for path in file_paths:
        if not path or not os.path.exists(path):
            print(f"⚠️  Missing file: {path}")
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for raw in f:
                    line = raw.strip()
                    if not _is_seq_line(line):
                        continue
                    s = line.split()[-1].upper()
                    s = ''.join([c for c in s if c in VALID_AA])
                    if len(s) >= 1:
                        s = s[:max_len_cap-2]   # -2 for SOS/EOS
                        seqs.append(list(s))
        except Exception as e:
            print(f"⚠️  Failed to read {path}: {e}")
    # dedupe
    uniq, seen = [], set()
    for lst in seqs:
        ss = ''.join(lst)
        if ss not in seen:
            seen.add(ss); uniq.append(lst)
    return uniq

def length_filter_and_upsample(uniq: List[List[str]],
                               keep_window: Tuple[int,int],
                               target_len: int,
                               near_delta: int,
                               near_target_min: int,
                               seed: int = 123) -> List[List[str]]:
    lo, hi = keep_window
    uniq = [s for s in uniq if lo <= len(s) <= hi]
    # upsample near target_len±delta
    near = [s for s in uniq if target_len-near_delta <= len(s) <= target_len+near_delta]
    rng = random.Random(seed)
    added = 0
    base = list(uniq)
    if len(near) > 0 and len(near) < near_target_min:
        need = near_target_min - len(near)
        base += [rng.choice(near) for _ in range(need)]
        added = need
    print(f"Upsampled near {target_len}±{near_delta}: +{added} (final {len(base)})")
    return base

# ========================= MODEL/PACKING SHIMS ===============================
class DataProcessor:
    """Minimal shim: αρκεί για unpickle + vocab + packing."""
    def sequence_to_indices(self, seq):
        vocab = getattr(self, "vocab", {}) or {}
        unk = vocab.get("<UNK>", 0)
        return np.asarray([vocab.get(tok, unk) for tok in seq], dtype=np.int32)

def pad_and_pack(processor, seq_chars: List[str], max_len: int):
    sos=processor.vocab['<SOS>']; eos=processor.vocab['<EOS>']; pad=processor.vocab['<PAD>']
    seq = ['<SOS>'] + list(seq_chars)[:max_len-2] + ['<EOS>']
    idx = processor.sequence_to_indices(seq)
    if len(idx) < max_len:
        idx = np.concatenate([idx, np.full((max_len-len(idx),), pad, np.int32)])
    return idx

def build_arrays(processor, seq_lists: List[List[str]], max_len: int):
    if len(seq_lists) == 0:
        return np.empty((0, max_len), dtype=np.int32)
    return np.asarray([pad_and_pack(processor, s, max_len) for s in seq_lists], dtype=np.int32)

def set_pad_identity(params, pad_idx: int):
    mps_enc = params['encoder']['mps']
    mps_dec = params['decoder']['mps']
    L, V, chi, _ = mps_enc.shape
    I = jnp.eye(chi, dtype=mps_enc.dtype)
    params['encoder']['mps'] = mps_enc.at[:, pad_idx].set(I)
    params['decoder']['mps'] = mps_dec.at[:, pad_idx].set(I)

def grow_params(params, old_len, new_len, bond_dim_mps, bond_dim_mera, vocab_size, rng):
    assert new_len > old_len and new_len % 2 == 0
    def grow_block(block, rng_block):
        mps_old = block['mps']; isos_old = block['isos']
        add_L = new_len - old_len
        k1, k2 = jax.random.split(rng_block)
        mps_new = jnp.zeros((new_len, vocab_size, bond_dim_mps, bond_dim_mps), dtype=mps_old.dtype)
        mps_new = mps_new.at[:old_len].set(mps_old * 0.999)
        if add_L > 0:
            init_tail = jax.random.normal(k1, (add_L, vocab_size, bond_dim_mps, bond_dim_mps)) * 0.1
            mps_new = mps_new.at[old_len:].set(init_tail)
        old_d = old_len // 2; new_d = new_len // 2
        isos_new = jnp.zeros((new_d, bond_dim_mps, bond_dim_mps, bond_dim_mera), dtype=isos_old.dtype)
        isos_new = isos_new.at[:old_d].set(isos_old * 0.999)
        if new_d - old_d > 0:
            init_iso_tail = jax.random.normal(k2, (new_d - old_d, bond_dim_mps, bond_dim_mps, bond_dim_mera)) * 0.1
            isos_new = isos_new.at[old_d:].set(init_iso_tail)
        return {'mps': mps_new, 'isos': isos_new}
    rng_enc, rng_dec, kproj = jax.random.split(rng, 3)
    new_encoder = grow_block(params['encoder'], rng_enc)
    new_decoder = grow_block(params['decoder'], rng_dec)
    old_td = (old_len//2) * bond_dim_mera
    new_td = (new_len//2) * bond_dim_mera
    proj_old = params['projection_matrix']
    proj_new = jnp.zeros((new_td, bond_dim_mps), dtype=proj_old.dtype)
    k = min(old_td, new_td)
    proj_new = proj_new.at[:k, :].set(proj_old[:k, :])
    if new_td > k:
        init_tail = jax.random.normal(kproj, (new_td - k, bond_dim_mps)) * 0.1
        proj_new = proj_new.at[k:, :].set(init_tail)
    return {
        'encoder': new_encoder,
        'decoder': new_decoder,
        'projection_matrix': proj_new,
        'output_projection': params['output_projection'],
    }

def shrink_params(params, old_len, new_len):
    assert new_len < old_len and new_len % 2 == 0
    def shrink_block(block):
        mps_old = block['mps']; isos_old = block['isos']
        return {'mps': mps_old[:new_len], 'isos': isos_old[:(new_len//2)]}
    new_encoder = shrink_block(params['encoder'])
    new_decoder = shrink_block(params['decoder'])
    old_td = (old_len//2) * params['encoder']['isos'].shape[-1]
    new_td = (new_len//2) * new_encoder['isos'].shape[-1]
    proj_old = params['projection_matrix']
    proj_new = proj_old[:new_td]
    return {
        'encoder': new_encoder,
        'decoder': new_decoder,
        'projection_matrix': proj_new,
        'output_projection': params['output_projection'],
    }

# ========================= ENCODE/DECODE (compact) ===========================
def _encode_step(carry, token_and_pos):
    state, mps, pos, rng_key, rate = carry
    tok, p = token_and_pos
    W = mps[p, tok]
    state = state @ W
    key = jax.random.fold_in(rng_key, p)
    keep = 1.0 - rate
    mask = jax.random.bernoulli(key, p=jnp.clip(keep, 0.0, 1.0), shape=state.shape)
    state = jnp.where(mask, state / jnp.maximum(keep, 1e-6), 0.0)
    return (state, mps, p+1, rng_key, rate), state

@jax.jit
def encode_compact(enc_params, seq_indices, rng_key, dropout_rate):
    L = enc_params['mps'].shape[0]
    chi = enc_params['mps'].shape[-1]
    state0 = jnp.zeros((chi,), dtype=jnp.float32).at[0].set(1.0)
    positions = jnp.arange(L, dtype=jnp.int32)
    (_, _, _, _, _), states = lax.scan(
        _encode_step,
        (state0, enc_params['mps'], 0, rng_key, dropout_rate),
        (seq_indices, positions)
    )
    def pair_reduce(_, i):
        l = states[2*i]; r = states[2*i+1]
        iso = enc_params['isos'][i]
        hi  = jnp.einsum('i,j,ijk->k', l, r, iso)
        return None, hi
    _, hi_list = lax.scan(pair_reduce, None, jnp.arange(L//2))
    tv = hi_list.reshape(-1)
    return tv / (jnp.linalg.norm(tv) + 1e-9)

def _decode_step(carry, tok_and_pos):
    vec, ctx, mps_dec, out_proj, first_flag = carry
    tok, p = tok_and_pos
    W = mps_dec[p, tok]
    vec = vec @ W
    vec = jnp.where(first_flag, vec + ctx, vec)
    logits = vec @ out_proj
    return (vec, ctx, mps_dec, out_proj, False), logits

@jax.jit
def decode_tf_compact(params, thought_vector, decoder_input):
    L = params['decoder']['mps'].shape[0]
    chi = params['decoder']['mps'].shape[-1]
    ctx = thought_vector @ params['projection_matrix']
    vec0 = jnp.zeros((chi,), dtype=jnp.float32).at[0].set(1.0)
    positions = jnp.arange(L-1, dtype=jnp.int32)
    mps_dec = params['decoder']['mps'][:-1]
    (_, _, _, _, _), logits = lax.scan(
        _decode_step,
        (vec0, ctx, mps_dec, params['output_projection'], True),
        (decoder_input, positions)
    )
    return logits  # (L-1, V)

# ========================= LOSS / TRAIN ======================================
def _xent_logits_with_int_labels(logits, labels_int, label_smoothing=0.0):
    V = logits.shape[-1]
    ls = jnp.asarray(label_smoothing, dtype=logits.dtype)
    onehot = jax.nn.one_hot(labels_int, V)
    smoothed = optax.smooth_labels(onehot, ls)
    return optax.softmax_cross_entropy(logits=logits, labels=smoothed)

@jax.jit
def per_seq_losses(params, batch_inputs, pad_idx, rng_key, dropout_rate, label_smoothing):
    B = batch_inputs.shape[0]
    enc_keys = jax.random.split(rng_key, B)
    tvs = jax.vmap(encode_compact, in_axes=(None,0,0,None))(
        params['encoder'], batch_inputs, enc_keys, dropout_rate
    )
    dec_in  = batch_inputs[:, :-1]
    dec_tgt = batch_inputs[:, 1:]
    logits  = jax.vmap(decode_tf_compact, in_axes=(None,0,0))(
        params, tvs, dec_in
    )  # (B,L-1,V)
    token_losses = _xent_logits_with_int_labels(logits, dec_tgt, label_smoothing)
    mask = (dec_tgt != pad_idx)
    per_seq_sum = (token_losses * mask).sum(axis=1)
    per_seq_cnt = mask.sum(axis=1)
    return per_seq_sum / jnp.maximum(per_seq_cnt, 1)

@jax.jit
def loss_fn(params, batch_inputs, pad_idx, rng_key, dropout_rate, label_smoothing):
    losses = per_seq_losses(params, batch_inputs, pad_idx, rng_key, dropout_rate, label_smoothing)
    counts = (batch_inputs[:, 1:] != pad_idx).sum(axis=1).astype(jnp.float32)
    denom = jnp.maximum(counts.sum(), 1.0)
    return (losses * counts).sum() / denom

def compute_batch_size(max_len: int) -> int:
    return max(8, min(BATCH_MAX, max(1, TOKENS_PER_BATCH // max(1, max_len))))

def make_optimizer(lr: float):
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr, weight_decay=2e-4)
    )

def compute_perplexity(params, data_arr, pad_idx, batch=64):
    n = len(data_arr)
    if n == 0:
        return float('inf')
    batch = max(1, min(batch, n))
    rng = jax.random.PRNGKey(999)
    tot_ll = 0.0; tot_tokens = 0
    for i in range(0, n, batch):
        chunk_np = np.asarray(data_arr[i:i+batch])
        chunk = jnp.asarray(chunk_np)
        l = float(loss_fn(params, chunk, pad_idx, rng, 0.0, 0.0))
        nonpad = int((chunk_np[:, 1:] != pad_idx).sum())
        tot_ll += l * nonpad
        tot_tokens += nonpad
    if tot_tokens == 0:
        return float('inf')
    return float(np.exp(tot_ll / tot_tokens))

# ---------- train steps ----------
def make_train_step_freeze_encoder(pad_idx, optimizer):
    @jax.jit
    def step(params, opt_state, batch, rng_key, dropout_rate, ls):
        l, grads = jax.value_and_grad(loss_fn)(
            params, batch, pad_idx, rng_key, dropout_rate, ls
        )
        grads = dict(grads)
        grads['encoder'] = jtu.tree_map(jnp.zeros_like, grads['encoder'])
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, l
    return step

def make_train_step_unfreeze(pad_idx, optimizer):
    @jax.jit
    def step(params, opt_state, batch, rng_key, dropout_rate, ls):
        l, grads = jax.value_and_grad(loss_fn)(
            params, batch, pad_idx, rng_key, dropout_rate, ls
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, l
    return step

# ========================= MAIN =============================================
def _evenize(n: int) -> int:
    return n if (n % 2 == 0) else (n - 1 if n > 1 else 2)

def _auto_target_from_window(win: Tuple[int,int]) -> int:
    mid = int(round(0.5*(win[0] + win[1])))
    return max(2, _evenize(mid))

def main():
    random.seed(PY_SEED); np.random.seed(PY_SEED)

    window = KEEP_WINDOW
    if window[0] < 1 or window[1] < window[0]:
        raise ValueError("Invalid KEEP_WINDOW.")
    tgt_len = TARGET_LEN if TARGET_LEN is not None else _auto_target_from_window(window)
    if tgt_len % 2 != 0:
        tgt_len = _evenize(tgt_len)
    if not (window[0] <= tgt_len <= window[1]):
        print(f"ℹ️ TARGET_LEN ({tgt_len}) not inside KEEP_WINDOW {window}. Clamping.")
        tgt_len = max(window[0], min(window[1], tgt_len))
        tgt_len = _evenize(tgt_len)

    if not os.path.exists(BASE_MODEL):
        raise FileNotFoundError(f"Base model not found: {BASE_MODEL}")
    with open(BASE_MODEL, "rb") as f:
        params, processor, model_len = pickle.load(f)
    if not is_compact(params):
        raise RuntimeError("Only compact param format supported.")

    chi_mps = int(params['encoder']['mps'].shape[-1])
    chim    = int(params['encoder']['isos'].shape[-1])
    vocab   = int(params['encoder']['mps'].shape[1])
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ▶ Loaded base ✓  len={model_len}  χ={chi_mps}  chim={chim}  vocab={vocab}")
    print(f"Range window = {window} | TARGET_LEN = {tgt_len}")

    # optional capacity bump
    if BOND_DIM_MPS_TUNED is not None or BOND_DIM_MERA_TUNED is not None:
        def maybe_bump_capacity(params, bond_dim_mps_new=None, bond_dim_mera_new=None, rng=None):
            mps = params['encoder']['mps']; isos = params['encoder']['isos']
            L, V, chi_old, _ = mps.shape
            chim_old = isos.shape[-1]
            chi_new  = bond_dim_mps_new  if bond_dim_mps_new  is not None else chi_old
            chim_new = bond_dim_mera_new if bond_dim_mera_new is not None else chim_old
            if chi_new == chi_old and chim_new == chim_old:
                return params
            def grow_block(block, k1, k2):
                mps_old = block['mps']; isos_old = block['isos']
                L, V, chi_old, _ = mps_old.shape
                chim_old = isos_old.shape[-1]
                mps_new = jnp.zeros((L, V, chi_new, chi_new), dtype=mps_old.dtype)
                isos_new = jnp.zeros((L//2, chi_new, chi_new, chim_new), dtype=isos_old.dtype)
                mps_new = mps_new.at[..., :chi_old, :chi_old].set(mps_old)
                isos_new = isos_new.at[..., :chi_old, :chi_old, :chim_old].set(isos_old)
                if chi_new > chi_old:
                    init_tail = jax.random.normal(k1, (L, V, chi_new-chi_old, chi_new-chi_old)) * 0.05
                    mps_new = mps_new.at[..., chi_old:, chi_old:].set(init_tail)
                if chim_new > chim_old:
                    init_iso_tail = jax.random.normal(k2, (L//2, chi_new, chi_new, chim_new-chim_old)) * 0.05
                    isos_new = isos_new.at[..., :, :, chim_old:].set(init_iso_tail)
                return {'mps': mps_new, 'isos': isos_new}
            k1, k2 = jax.random.split(jax.random.PRNGKey(42))
            new_encoder = grow_block(params['encoder'], k1, k2)
            new_decoder = grow_block(params['decoder'], k1, k2)
            old_td = (new_encoder['mps'].shape[0]//2) * chim
            new_td = (new_encoder['mps'].shape[0]//2) * new_encoder['isos'].shape[-1]
            proj_old = params['projection_matrix']
            proj_new = jnp.zeros((new_td, new_encoder['mps'].shape[-1]), dtype=proj_old.dtype)
            kk = min(old_td, new_td); cc = min(proj_old.shape[1], new_encoder['mps'].shape[-1])
            proj_new = proj_new.at[:kk, :cc].set(proj_old[:kk, :cc])
            out_old = params['output_projection']
            out_new = jnp.zeros((new_encoder['mps'].shape[-1], out_old.shape[1]), dtype=out_old.dtype).at[:cc,:].set(out_old[:cc,:])
            return {'encoder': new_encoder, 'decoder': new_decoder,
                    'projection_matrix': proj_new, 'output_projection': out_new}
        params = maybe_bump_capacity(params, BOND_DIM_MPS_TUNED, BOND_DIM_MERA_TUNED, jax.random.PRNGKey(42))
        chi_mps = int(params['encoder']['mps'].shape[-1])
        chim    = int(params['encoder']['isos'].shape[-1])

    # build candidates
    cands = read_pfam_sequences(PFAM_FILES, max_len_cap=256)
    print(f"Raw candidates: {len(cands)} — filtering to window {window} and upsampling near L={tgt_len}…")
    uniq = length_filter_and_upsample(
        cands, keep_window=window, target_len=tgt_len,
        near_delta=NEAR_DELTA, near_target_min=NEAR_TARGET_MIN, seed=123
    )

    # prepare arrays @ tgt_len
    if model_len != tgt_len:
        if model_len > tgt_len:
            print(f"Shrinking model parameters: {model_len} → {tgt_len}")
            params = shrink_params(params, model_len, tgt_len)
        else:
            print(f"Growing model parameters: {model_len} → {tgt_len}")
            params = grow_params(params, model_len, tgt_len, chi_mps, chim, vocab, jax.random.PRNGKey(99))
        model_len = tgt_len

    pad_idx = processor.vocab['<PAD>']
    set_pad_identity(params, pad_idx)

    rng = random.Random(1337)
    rng.shuffle(uniq)
    n_total = len(uniq)
    if n_total == 0:
        print("⚠️ No data after filters. Consider widening KEEP_WINDOW.")
        return
    n_val = max(200, int(0.2 * n_total))
    train_seqs = uniq[:-n_val]
    val_seqs   = uniq[-n_val:]

    train_arr = build_arrays(processor, train_seqs, model_len)
    val_arr   = build_arrays(processor, val_seqs,   model_len)
    bs = compute_batch_size(model_len)
    print(f"Filtered unique={n_total} | train={len(train_arr)} val={len(val_arr)} | batch≈{bs}")

    # warm-up JIT
    if len(train_arr) >= 1:
        dummy = jnp.asarray(train_arr[:max(1, min(8, len(train_arr)))])
        _ = loss_fn(params, dummy, pad_idx, jax.random.PRNGKey(7), 0.0, 0.0).block_until_ready()

    # -------- Phase A: freeze encoder --------
    optimizer_a   = make_optimizer(LR_A)
    opt_state     = optimizer_a.init(params)
    train_step_A  = make_train_step_freeze_encoder(pad_idx, optimizer_a)
    best_val = float('inf'); best_params = params; no_imp = 0
    rng_key = jax.random.PRNGKey(0)

    for epoch in range(1, EPOCHS_A+1):
        t0 = time.time()
        perm = np.random.permutation(len(train_arr))
        tot = 0.0; nb=0; total_tokens=0
        for i in range(0, len(train_arr), bs):
            idx = perm[i:i+bs]
            if len(idx) == 0: continue
            batch = jnp.asarray(train_arr[idx])
            rng_key, subk = jax.random.split(rng_key)
            params, opt_state, loss = train_step_A(params, opt_state, batch, subk, DROPOUT_A, SMOOTH_A)
            tot += float(loss); nb += 1
            total_tokens += int((np.asarray(train_arr[idx])[:,1:] != pad_idx).sum())
        train_ppl = float(np.exp(tot/nb)) if nb>0 else float('inf')
        val_ppl = compute_perplexity(params, val_arr, pad_idx, batch=min(bs, len(val_arr)))
        dt = time.time()-t0
        print(f"[A] {epoch:03d}/{EPOCHS_A} | train {train_ppl:.4f} | val {val_ppl:.4f} | {dt:.1f}s")
        if val_ppl + 1e-3 < best_val:
            best_val = val_ppl; best_params = params; no_imp = 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE_A:
                print("Early stop Phase A.")
                break
    params = best_params

    # -------- Phase B: unfreeze all --------
    optimizer_b   = make_optimizer(LR_B)
    opt_state     = optimizer_b.init(params)
    train_step_B  = make_train_step_unfreeze(pad_idx, optimizer_b)
    best_val = float('inf'); best_params = params; no_imp = 0
    rng_key = jax.random.PRNGKey(1234)

    for epoch in range(1, EPOCHS_B+1):
        t0 = time.time()
        perm = np.random.permutation(len(train_arr))
        tot = 0.0; nb=0
        for i in range(0, len(train_arr), bs):
            idx = perm[i:i+bs]
            if len(idx) == 0: continue
            batch = jnp.asarray(train_arr[idx])
            rng_key, subk = jax.random.split(rng_key)
            params, opt_state, loss = train_step_B(params, opt_state, batch, subk, DROPOUT_B, SMOOTH_B)
            tot += float(loss); nb += 1
        train_ppl = float(np.exp(tot/nb)) if nb>0 else float('inf')
        val_ppl = compute_perplexity(params, val_arr, pad_idx, batch=min(bs, len(val_arr)))
        dt = time.time()-t0
        print(f"[B] {epoch:03d}/{EPOCHS_B} | train {train_ppl:.4f} | val {val_ppl:.4f} | {dt:.1f}s")
        if val_ppl + 1e-3 < best_val:
            best_val = val_ppl; best_params = params; no_imp = 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE_B:
                print("Early stop Phase B.")
                break
    params = best_params

    # save
    with open(TUNED_MODEL, "wb") as f:
        pickle.dump((params, processor, model_len), f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(TUNED_MODEL)/1024/1024
    print(f"✅ Fine-tune done. Saved best → {TUNED_MODEL} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    main()
