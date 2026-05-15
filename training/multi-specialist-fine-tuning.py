# -*- coding: utf-8 -*-
# ==============================================================================
#   Multi-Specialist Fine-Tuning (from MASTER tuned)
#   - Loads a single tuned base (compact MERA-MPS)
#   - Per window [Lmin,Lmax]:
#       * filter Pfam within window + upsample near anchor (±Δ)
#       * set specialist MAX_LEN at anchor (even)
#       * Phase A: freeze encoder + (optionally) head
#       * Phase B: unfreeze encoder; head usually frozen for calibration
#   - Saves one specialist .pkl per window + a JSON manifest (incl. best val PPL)
#   - Memory-safe (JAX preallocate OFF)
# ==============================================================================

import os, re, time, math, json, random, pickle, warnings
from typing import List, Tuple, Optional

# ---- JAX memory safety (set BEFORE importing jax) ----
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

# ========================== USER SETTINGS ====================================
BASE_MODEL   = os.environ.get("MASTER_TUNED", "/kaggle/input/alltrainedmodels/other/default/1/production_model_tuned.pkl")
OUT_DIR      = os.environ.get("SPECIALISTS_DIR", "/kaggle/working/specialists")
os.makedirs(OUT_DIR, exist_ok=True)

PFAM_FILES   = [
    "/kaggle/input/pfaseed/Pfam-A.seed",
    "/kaggle/input/pfafull/Pfam-A.full",
]

# --- Specialist windows (καλύπτουν καλά AMPs/short peptides)
SPECIALIST_WINDOWS: List[Tuple[int,int]] = [
    (24, 36),   # anchor ≈ 30/32
    (30, 48),   # anchor ≈ 40
    (44, 60),   # anchor ≈ 52/54
]

# Anchor selection for specialist MAX_LEN (must be even): "center_even" | "max_even"
ANCHOR_POLICY = "center_even"

# Dataset shaping inside each window
NEAR_DELTA             = 2       # upsample γύρω από anchor±Δ
NEAR_TARGET_MIN        = 1600    # ελάχιστα δείγματα κοντά στο anchor (με repeats)
HARD_CAP_MAX_LEN       = 256     # parser cap (SOS/EOS safety)
MIN_SAMPLES_REQUIRED   = 800     # αν < skip το window (για σταθερό training)

# Training global
TOKENS_PER_BATCH  = 8192
BATCH_MAX         = 64
PY_SEED           = 0

# Phase A (freeze encoder + head)
EPOCHS_A   = 10
LR_A       = 1.0e-4
DROPOUT_A  = 0.10
SMOOTH_A   = 0.02
PATIENCE_A = 4
FREEZE_HEAD_A = True   # shared head → σταθερή κλίμακα

# Phase B (unfreeze encoder); head συμπεριφορά
EPOCHS_B   = 100
LR_B       = 8.0e-5
DROPOUT_B  = 0.10
SMOOTH_B   = 0.02
PATIENCE_B = 8
FREEZE_HEAD_B = True   # προτείνεται TRUE για calibration consistency
HEAD_L2_B  = 0.0       # αν ξεκλειδώσεις head, βάλε π.χ. 1e-4

# ========================= SANITY / HELPERS ==================================
def is_compact(params):
    try:
        return ('encoder' in params and 'decoder' in params and
                'projection_matrix' in params and 'output_projection' in params and
                isinstance(params['encoder'], dict) and isinstance(params['decoder'], dict))
    except Exception:
        return False

def even(x: int) -> int:
    return x if x % 2 == 0 else (x+1)

def anchor_for_window(lo: int, hi: int, policy: str="center_even") -> int:
    if policy == "max_even":
        return even(hi)
    mid = (lo + hi) // 2
    return even(mid)

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

def filter_window(uniq: List[List[str]], lo: int, hi: int) -> List[List[str]]:
    return [s for s in uniq if lo <= len(s) <= hi]

def upsample_near_anchor(seqs: List[List[str]], anchor: int, delta: int, min_near: int, seed: int=123) -> List[List[str]]:
    rng = random.Random(seed)
    near = [s for s in seqs if (anchor - delta) <= len(s) <= (anchor + delta)]
    added = 0
    if len(near) > 0 and len(near) < min_near:
        need = min_near - len(near)
        seqs = list(seqs) + [rng.choice(near) for _ in range(need)]
        added = need
    print(f"  ↪ upsample near {anchor}±{delta}: +{added} (final {len(seqs)})")
    return seqs

# ========================= PACKING ===========================================
class DataProcessor:
    """Minimal shim for unpickling + vocab access, kept from base checkpoint."""
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

# ========================= MODEL SHAPE OPS ===================================
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
    return logits

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
    logits  = jax.vmap(decode_tf_compact, in_axes=(None,0,0))(params, tvs, dec_in)
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

def make_optimizer(lr: float, head_l2: float = 0.0):
    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr, weight_decay=2e-4)
    )
    return opt

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
def make_train_step_freeze(pad_idx, optimizer, freeze_encoder=True, freeze_head=True, head_l2=0.0):
    @jax.jit
    def step(params, opt_state, batch, rng_key, dropout_rate, ls):
        l, grads = jax.value_and_grad(loss_fn)(
            params, batch, pad_idx, rng_key, dropout_rate, ls
        )
        if head_l2 > 0.0 and not freeze_head:
            l2 = 0.5 * head_l2 * jnp.sum(params['output_projection']**2)
            l = l + l2
            g_head = grads['output_projection'] + head_l2 * params['output_projection']
            grads = dict(grads); grads['output_projection'] = g_head
        if freeze_encoder:
            grads = dict(grads); grads['encoder'] = jtu.tree_map(jnp.zeros_like, grads['encoder'])
        if freeze_head:
            grads = dict(grads); grads['output_projection'] = jnp.zeros_like(grads['output_projection'])
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, l
    return step

def make_train_step_unfreeze(pad_idx, optimizer, freeze_head=True, head_l2=0.0):
    @jax.jit
    def step(params, opt_state, batch, rng_key, dropout_rate, ls):
        l, grads = jax.value_and_grad(loss_fn)(
            params, batch, pad_idx, rng_key, dropout_rate, ls
        )
        if head_l2 > 0.0 and not freeze_head:
            l2 = 0.5 * head_l2 * jnp.sum(params['output_projection']**2)
            l = l + l2
            grads = dict(grads)
            grads['output_projection'] = grads['output_projection'] + head_l2 * params['output_projection']
        if freeze_head:
            grads = dict(grads)
            grads['output_projection'] = jnp.zeros_like(grads['output_projection'])
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, l
    return step

# ========================= MULTI-FINETUNE LOOP ===============================
def main():
    def now(): return time.strftime("%Y-%m-%d %H:%M:%S")
    random.seed(PY_SEED); np.random.seed(PY_SEED)

    if not os.path.exists(BASE_MODEL):
        raise FileNotFoundError(f"Base model not found: {BASE_MODEL}")
    with open(BASE_MODEL, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, tuple) and len(payload) >= 3:
        base_params, processor, base_len = payload[:3]
    else:
        raise RuntimeError("Unsupported base model pickle format.")
    if not is_compact(base_params):
        raise RuntimeError("Only compact param format supported.")

    chi_mps = int(base_params['encoder']['mps'].shape[-1])
    chim    = int(base_params['encoder']['isos'].shape[-1])
    vocab   = int(base_params['encoder']['mps'].shape[1])
    print(f"▶ Loaded MASTER ✓  len={base_len}  χ={chi_mps}  chim={chim}  vocab={vocab}")

    all_seqs = read_pfam_sequences(PFAM_FILES, max_len_cap=HARD_CAP_MAX_LEN)
    print(f"Candidates (deduped): {len(all_seqs)}")

    manifest = {"base_model": BASE_MODEL, "specialists": []}

    for (Lmin, Lmax) in SPECIALIST_WINDOWS:
        Lmin_i, Lmax_i = int(Lmin), int(Lmax)
        L_anchor = anchor_for_window(Lmin_i, Lmax_i, ANCHOR_POLICY)
        if L_anchor % 2 != 0: L_anchor += 1
        if L_anchor < 2: L_anchor = 2

        # track best val PPL per phase for manifest
        best_val_A = None
        best_val_B = None

        print("\n" + "="*74)
        print(f"  Specialist window [{Lmin_i}, {Lmax_i}]  | anchor L={L_anchor} (policy: {ANCHOR_POLICY})")
        print("="*74)

        seqs = filter_window(all_seqs, Lmin_i, Lmax_i)
        print(f"  raw in-window: {len(seqs)}")
        seqs = upsample_near_anchor(seqs, L_anchor, NEAR_DELTA, NEAR_TARGET_MIN, seed=123)
        if len(seqs) < MIN_SAMPLES_REQUIRED:
            print(f"  ⚠️  Too few samples ({len(seqs)}) — skipping this window.")
            continue

        rng = random.Random(1337 + L_anchor)
        rng.shuffle(seqs)
        n_val = max(200, int(0.2 * len(seqs)))
        train_seqs = seqs[:-n_val]
        val_seqs   = seqs[-n_val:]

        params = pickle.loads(pickle.dumps(base_params, protocol=pickle.HIGHEST_PROTOCOL))
        model_len = int(params['encoder']['mps'].shape[0])

        if model_len != L_anchor:
            if model_len > L_anchor:
                print(f"  Shrinking model parameters: {model_len} → {L_anchor}")
                params = shrink_params(params, model_len, L_anchor)
            else:
                print(f"  Growing model parameters: {model_len} → {L_anchor}")
                params = grow_params(params, model_len, L_anchor, chi_mps, chim, vocab, jax.random.PRNGKey(99))
            model_len = L_anchor

        pad_idx = processor.vocab['<PAD>']
        set_pad_identity(params, pad_idx)

        train_arr = build_arrays(processor, train_seqs, model_len)
        val_arr   = build_arrays(processor, val_seqs,   model_len)
        bs = compute_batch_size(model_len)
        print(f"  window unique={len(seqs)} | train={len(train_arr)} val={len(val_arr)} | batch≈{bs}")

        # Warmup JIT
        if len(train_arr) >= 1:
            dummy = jnp.asarray(train_arr[:max(1, min(8, len(train_arr)))])
            _ = loss_fn(params, dummy, pad_idx, jax.random.PRNGKey(7), 0.0, 0.0).block_until_ready()

        # ---------- Phase A ----------
        if EPOCHS_A > 0:
            optimizer_a  = make_optimizer(LR_A)
            opt_state    = optimizer_a.init(params)
            step_A       = make_train_step_freeze(
                pad_idx, optimizer_a,
                freeze_encoder=True,
                freeze_head=FREEZE_HEAD_A,
                head_l2=0.0
            )
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
                    params, opt_state, loss = step_A(params, opt_state, batch, subk, DROPOUT_A, SMOOTH_A)
                    tot += float(loss); nb += 1
                    total_tokens += int((np.asarray(train_arr[idx])[:,1:] != pad_idx).sum())
                train_ppl = float(np.exp(tot/nb)) if nb>0 else float('inf')
                val_ppl = compute_perplexity(params, val_arr, pad_idx, batch=min(bs, len(val_arr)))
                dt = time.time()-t0
                print(f"  [A] {epoch:03d}/{EPOCHS_A} | train {train_ppl:.4f} | val {val_ppl:.4f} | {dt:.1f}s")
                if val_ppl + 1e-3 < best_val:
                    best_val = val_ppl; best_params = params; no_imp = 0
                else:
                    no_imp += 1
                    if no_imp >= PATIENCE_A:
                        print("  Early stop Phase A.")
                        break
            best_val_A = float(best_val)
            params = best_params

        # ---------- Phase B ----------
        if EPOCHS_B > 0:
            optimizer_b  = make_optimizer(LR_B, head_l2=HEAD_L2_B)
            opt_state    = optimizer_b.init(params)
            step_B       = make_train_step_unfreeze(
                pad_idx, optimizer_b,
                freeze_head=FREEZE_HEAD_B,
                head_l2=HEAD_L2_B
            )
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
                    params, opt_state, loss = step_B(params, opt_state, batch, subk, DROPOUT_B, SMOOTH_B)
                    tot += float(loss); nb += 1
                train_ppl = float(np.exp(tot/nb)) if nb>0 else float('inf')
                val_ppl = compute_perplexity(params, val_arr, pad_idx, batch=min(bs, len(val_arr)))
                dt = time.time()-t0
                print(f"  [B] {epoch:03d}/{EPOCHS_B} | train {train_ppl:.4f} | val {val_ppl:.4f} | {dt:.1f}s")
                if val_ppl + 1e-3 < best_val:
                    best_val = val_ppl; best_params = params; no_imp = 0
                else:
                    no_imp += 1
                    if no_imp >= PATIENCE_B:
                        print("  Early stop Phase B.")
                        break
            best_val_B = float(best_val)
            params = best_params

        # ---------- Save specialist ----------
        tag = f"win{int(Lmin)}-{int(Lmax)}_len{int(L_anchor)}"
        out_path = os.path.join(OUT_DIR, f"specialist_{tag}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump((params, processor, int(L_anchor)), f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = os.path.getsize(out_path)/1024/1024
        print(f"  ✅ Saved specialist → {out_path} ({size_mb:.2f} MB)")

        # Επιλογή καλύτερου validation perplexity για το manifest
        val_ppl_best = best_val_B if (best_val_B is not None) else best_val_A
        if val_ppl_best is None:
            val_ppl_best = float("nan")

        manifest["specialists"].append({
            "window": [int(Lmin), int(Lmax)],
            "anchor_len": int(L_anchor),
            "path": out_path,
            "val_ppl_best": float(val_ppl_best),
            "freeze_head_A": bool(FREEZE_HEAD_A),
            "freeze_head_B": bool(FREEZE_HEAD_B),
            "head_L2_B": float(HEAD_L2_B),
            "epochs_A": int(EPOCHS_A),
            "epochs_B": int(EPOCHS_B),
            "near_delta": int(NEAR_DELTA),
            "near_target_min": int(NEAR_TARGET_MIN),
            "batch_tokens": int(TOKENS_PER_BATCH),
        })

    # Save manifest
    manifest_path = os.path.join(OUT_DIR, "specialists_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\n📄 Manifest saved: {manifest_path}")
    print("Done.")

if __name__ == "__main__":
    main()
