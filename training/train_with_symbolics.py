%%writefile /kaggle/working/train_with_symbolics.py
# -*- coding: utf-8 -*-

import os, time, pickle, random, re, math
import numpy as np
import jax, jax.numpy as jnp
from jax import lax
import optax
from typing import List, Tuple, Dict, Optional

from symbolics_integration import (
    SymbolicsHelper, ensure_symbolic_params,
    make_train_step_symbolic, make_train_step_symbolic_pmap, get_pairs_for_batch,
    DEFAULT_PROTEIN_CONFIG,
)

# ===================== TUNABLE SETTINGS (YOUR PRESET) =====================
DATA_FILES: List[str]   = ["/kaggle/input/pfaseed/Pfam-A.seed"]

# Data / curriculum
MIN_LEN_LIMIT: int      = 10             # inclusive lower bound for corpus filtering
MAX_LEN_LIMIT: int      = 52            # EVEN, <= FIXED_MAX_LEN_CAP   66   pame->22
FIXED_MAX_LEN_CAP: int  = 256           # EVEN, parsing hard cap
AUTO_MAX_STAGES: int    = 8
MIN_BUCKET_FOR_STAGE: int = 300
FRAC_MODE: str          = "uniform"     # "uniform" or "by_tokens"
NUM_SAMPLES: int        = 20000         # target per curriculum (selected across stages)

# Model
BOND_DIMENSION_MPS: int = 32   #32
BOND_DIMENSION_MERA:int = 32   #32

# Train (dynamic tokens-per-batch scheduling)
TOKENS_PER_BATCH_TARGET = 8192
BATCH_SIZE = 80     #64
MIN_STAGE_BATCH: int    = 8             # floor per stage


#TOKENS_PER_BATCH_TARGET = 4096
#BATCH_SIZE = 32

MIN_STAGE_BATCH: int    = 8             # floor per stage

BASE_LR: float          = 2e-4
WEIGHT_DECAY: float     = 2e-4
DROPOUT_RATE_TRAIN:float= 0.08
LABEL_SMOOTHING: float  = 0.03
WARMUP_FRAC: float      = 0.10     #0.10
COSINE_FINAL_RATIO:float= 0.1
EPOCHS_PER_STAGE: int   = 800
MIN_EPOCHS: int         = 400   #800
PATIENCE: int           = 8
MIN_DELTA: float        = 1e-3

# ---------- Minimum validation PPL requirement ----------
# Set e.g. 9.2 to keep training (with upsampling) until best val-PPL <= 9.2
MIN_VAL_PPL_REQUIRED: Optional[float] = 30

# ---------- Multi-GPU ----------
USE_PMAP_MULTI_GPU: bool = True
PMAP_AXIS_NAME: str = "dp"  # data-parallel axis

# ---------- Checkpoint cadence ----------
# Save 'latest' every N epochs. Set 0 to disable periodic epoch saves.
SAVE_LATEST_EVERY: int = 15
# Always save 'latest' on early stop (even if not on SAVE_LATEST_EVERY boundary)
SAVE_LATEST_ON_EARLY_STOP: bool = True
# Always save 'latest' at the end of each stage
SAVE_LATEST_ON_STAGE_END: bool = True

# NEW: Best checkpoint cadence controls
# If False, never write 'best' to disk during epochs (only stage-end/early-stop flush if enabled below).
SAVE_BEST: bool = True
# If 0 => never write 'best' during epochs (buffer only; flush at stage end / early stop).
# If 1 => write immediately on each improvement (old behavior).
# If N>1 => buffer improvements and write only on epochs that are multiples of N.
SAVE_BEST_EVERY: int = 20
# Force-flush buffered best at early stop / stage end:
FLUSH_BEST_ON_EARLY_STOP: bool = True
FLUSH_BEST_ON_STAGE_END: bool  = True

# Symbolics
USE_SYMBOLICS = True
GRAPH_EDGES_PER_NODE = 4
GRAPH_MAX_PAIRS = 2048
LAMBDA_GRAPH_BASE = 0.003   # start small and ramp

# Adaptive upsampling (Balanced-HEM)
ADAPTIVE_UPSAMPLING: bool = True
UPSAMPLE_TRIGGER_PATIENCE: int = 2   #4
MAX_UPSAMPLE_EVENTS_PER_STAGE: int = 6 #10
COOLDOWN_EPOCHS: int = 8   #80
REL_IMPROVEMENT_WINDOW: int = 8    #20
MIN_REL_IMPROVEMENT: float = 0.02    #0.02
PPL_ABS_CEILING: Optional[float] = None

UPSAMPLE_FACTOR_BASE: float = 1.30
UPSAMPLE_FACTOR_STEP: float = 0.10
MAX_UPSAMPLE_FACTOR: float = 1.80
REPEAT_CAP_MULTIPLIER: float = 2.0
UPSAMPLED_DROPOUT: float = 0.18
UPSAMPLED_SMOOTH: float = 0.08

# Hard-Example Mining
ENABLE_HEM: bool = True
HEM_ON_UPSAMPLE: bool = True
HEM_CANDIDATE_FRAC: float = 0.20
HEM_CANDIDATE_MAX: int = 8000
HEM_EVAL_TOKENS_PER_BATCH: int = 8192

# RNG / Paths
RNG_KEY = jax.random.PRNGKey(0)
PY_SEED = 0
CHECKPOINT_DIR = '/kaggle/working/'
MODEL_FILE     = '/kaggle/working/production_model_bio.pkl'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ===================== Data Processor =====================
class DataProcessor:
    def __init__(self, sequence_type='protein'):
        if sequence_type == 'protein':
            self.valid_chars = set('ACDEFGHIKLMNPQRSTVWY')
        elif sequence_type == 'dna':
            self.valid_chars = set('ACGT')
        elif sequence_type == 'rna':
            self.valid_chars = set('ACGU')
        else:
            raise ValueError("Unsupported sequence_type.")
        self.special_tokens = ['<PAD>','<SOS>','<EOS>','<UNK>']
        self.vocab, self.inv_vocab = {}, {}
        self.word_count = 0

    def build_vocab(self):
        for i, tok in enumerate(self.special_tokens):
            self.vocab[tok] = i; self.inv_vocab[i] = tok
        idx = len(self.special_tokens)
        for ch in sorted(list(self.valid_chars)):
            self.vocab[ch] = idx; self.inv_vocab[idx] = ch; idx += 1
        self.word_count = len(self.vocab)

    def _is_seq_line(self, line: str) -> bool:
        if (not line) or line.startswith('#') or line.startswith('//'):
            return False
        parts = line.split()
        if len(parts) < 2: return False
        seq = parts[-1]
        return bool(re.fullmatch(r'[A-Za-z\-\.]+', seq))

    def load_stockholm_ungapped(self, file_paths: List[str], min_len=1, max_len_cap=256):
        all_seqs: List[List[str]] = []
        for path in file_paths:
            if not os.path.exists(path):
                print(f"⚠️  File not found: {path}. Skipping.")
                continue
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for raw in f:
                        line = raw.strip()
                        if not self._is_seq_line(line):
                            continue
                        parts = line.split()
                        seq = parts[-1].upper()
                        seq = ''.join([c for c in seq if c not in ('.','-') and c in self.valid_chars])
                        if len(seq) >= min_len:
                            seq = seq[:max_len_cap-2]  # -2 for SOS/EOS
                            all_seqs.append(list(seq))
            except Exception as e:
                print(f"⚠️  Failed to read {path}: {e}")
                continue
        # deduplicate
        uniq, seen = [], set()
        for lst in all_seqs:
            s = ''.join(lst)
            if s not in seen:
                seen.add(s); uniq.append(lst)
        return uniq

    def sequence_to_indices(self, seq_chars: List[str]):
        unk = self.vocab['<UNK>']
        return np.array([self.vocab.get(c, unk) for c in seq_chars], dtype=np.int32)

# ===================== Model / Init =====================

def init_single_mera_compact(key, vocab_size, L, chi_mps, chi_mera):
    k_mps, k_iso = jax.random.split(key)
    mps = jax.random.normal(k_mps, (L, vocab_size, chi_mps, chi_mps)) * 0.1
    isos = jax.random.normal(k_iso, (L//2, chi_mps, chi_mps, chi_mera)) * 0.1
    return {"mps": mps, "isos": isos}

def init_encoder_decoder_model(key, physical_dim, bond_dim_mps, bond_dim_mera, max_len):
    assert max_len % 2 == 0, "MAX_LEN must be even for MERA pairwise reduction"
    k1,k2,k3,k4 = jax.random.split(key,4)
    enc = init_single_mera_compact(k1, physical_dim, max_len, bond_dim_mps, bond_dim_mera)
    dec = init_single_mera_compact(k2, physical_dim, max_len, bond_dim_mps, bond_dim_mera)
    thought_dim = (max_len//2) * bond_dim_mera
    projection_matrix = jax.random.normal(k3, (thought_dim, bond_dim_mps)) * 0.1
    output_projection = jax.random.normal(k4, (bond_dim_mps, physical_dim)) * 0.1
    return {
        'encoder': enc,
        'decoder': dec,
        'projection_matrix': projection_matrix,
        'output_projection': output_projection
    }

# ===================== Dropout helper =====================

def apply_dropout(x, key, rate):
    rate = jnp.asarray(rate, dtype=x.dtype)
    keep_prob = jnp.clip(1.0 - rate, 0.0, 1.0)
    def do_keep(_):
        mask = jax.random.bernoulli(key, p=keep_prob, shape=x.shape)
        return jnp.where(mask, x / jnp.maximum(keep_prob, 1e-6), 0.0)
    return lax.cond(jnp.equal(rate, 0.0), lambda _: x, do_keep, operand=None)

# ===================== Encode / Decode =====================

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
    assert L % 2 == 0, "Encoder length must be even for MERA pairwise reduction"
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

# ===================== Loss / Train =====================

@jax.jit
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
    dec_keys = jax.random.split(jax.random.fold_in(rng_key, 12345), B)
    tvs = jax.vmap(encode, in_axes=(None,0,0,None))(
        params['encoder'], batch_inputs, enc_keys, dropout_rate
    )
    dec_in  = batch_inputs[:, :-1]
    dec_tgt = batch_inputs[:, 1:]
    logits  = jax.vmap(decode_teacher_forcing, in_axes=(None,0,0,0,None))(
        params, tvs, dec_in, dec_keys, dropout_rate
    )  # (B,L-1,V)
    token_losses = _xent_logits_with_int_labels(logits, dec_tgt, label_smoothing)
    mask = (dec_tgt != pad_idx)
    per_seq_sum = (token_losses * mask).sum(axis=1)
    per_seq_cnt = mask.sum(axis=1)
    return per_seq_sum / jnp.maximum(per_seq_cnt, 1)

@jax.jit
def loss_fn(params, batch_inputs, batch_targets, pad_idx, rng_key, dropout_rate, label_smoothing):
    losses = per_seq_losses(params, batch_inputs, pad_idx, rng_key, dropout_rate, label_smoothing)
    counts = (batch_targets[:, 1:] != pad_idx).sum(axis=1).astype(jnp.float32)
    denom = jnp.maximum(counts.sum(), 1.0)
    return (losses * counts).sum() / denom

# ===================== Helpers =====================

def set_pad_identity(params, pad_idx):
    mps_enc = params['encoder']['mps']
    mps_dec = params['decoder']['mps']
    L, V, chi, _ = mps_enc.shape
    I = jnp.eye(chi, dtype=mps_enc.dtype)
    mps_enc = mps_enc.at[:, pad_idx].set(I)
    mps_dec = mps_dec.at[:, pad_idx].set(I)
    params['encoder']['mps'] = mps_enc
    params['decoder']['mps'] = mps_dec

def pad_and_pack(processor, seq_chars, max_len):
    sos=processor.vocab['<SOS>']; eos=processor.vocab['<EOS>']; pad=processor.vocab['<PAD>']
    seq = ['<SOS>'] + seq_chars[:max_len-2] + ['<EOS>']
    idx = processor.sequence_to_indices(seq)
    if len(idx) < max_len:
        idx = np.concatenate([idx, np.full((max_len-len(idx),), pad, np.int32)])
    return idx

def build_arrays(processor, seq_lists, max_len):
    if len(seq_lists) == 0:
        return np.empty((0, max_len), dtype=np.int32)
    return np.asarray([pad_and_pack(processor, s, max_len) for s in seq_lists], dtype=np.int32)

@jax.jit
def eval_loss_jit(params, data_arr, pad_idx, rng_key, label_smoothing):
    l = loss_fn(params, data_arr, data_arr, pad_idx, rng_key, 0.0, label_smoothing)
    return l

def compute_perplexity(params, data_arr, pad_idx, batch=64, label_smoothing=0.0):
    n = len(data_arr)
    if n == 0:
        return float('inf')
    batch = max(1, min(batch, n))
    rng = jax.random.PRNGKey(999)
    tot_ll = 0.0; tot_tokens = 0
    for i in range(0, n, batch):
        chunk_np = np.asarray(data_arr[i:i+batch])
        chunk = jnp.asarray(chunk_np)
        l = float(eval_loss_jit(params, chunk, pad_idx, rng, label_smoothing))
        nonpad = int((chunk_np[:, 1:] != pad_idx).sum())
        tot_ll += l * nonpad
        tot_tokens += nonpad
    if tot_tokens == 0:
        return float('inf')
    return float(np.exp(tot_ll / tot_tokens))

def atomic_pickle_save(obj, path):
    tmp = path + ".tmp"
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    with open(path, 'rb') as f:
        _ = pickle.load(f)

# ---------- Dataset persistence helpers ----------
def seq_key(chars_list: List[str]) -> str:
    return ''.join(chars_list)

def rebuild_stage_extra(all_sequences: List[List[str]], bounds: Tuple[int,int], used_keys_set: set):
    low, high = bounds
    candidates = [s for s in all_sequences if low <= len(s) <= high]
    extra = [s for s in candidates if seq_key(s) not in used_keys_set]
    return extra

# ---------- Checkpoint helpers ----------

def stage_ckpt_path(ckpt_dir: str, max_len: int) -> str:
    return os.path.join(ckpt_dir, f"latest_len{max_len}.pkl")

def best_stage_ckpt_path(ckpt_dir: str, max_len: int) -> str:
    return os.path.join(ckpt_dir, f"best_len{max_len}.pkl")

def try_load_checkpoint(path: str):
    try:
        with open(path, 'rb') as f:
            ch = pickle.load(f)
        return ch
    except Exception:
        return None

def load_checkpoint_for_len(ckpt_dir: str, max_len: int):
    stage_path = stage_ckpt_path(ckpt_dir, max_len)
    ch = try_load_checkpoint(stage_path)
    if ch is not None:
        return ch, stage_path
    generic = os.path.join(ckpt_dir, 'latest.pkl')
    ch = try_load_checkpoint(generic)
    if ch is not None:
        return ch, generic
    return None, None

def save_checkpoint_dual(obj, ckpt_dir: str, max_len: int, is_best: bool=False):
    generic = os.path.join(ckpt_dir, 'latest.pkl')
    stage   = stage_ckpt_path(ckpt_dir, max_len)
    atomic_pickle_save(obj, generic)
    atomic_pickle_save(obj, stage)
    if is_best:
        best_p = best_stage_ckpt_path(ckpt_dir, max_len)
        atomic_pickle_save(obj, best_p)
    size_mb = os.path.getsize(generic)/1024/1024
    tag = " + best" if is_best else ""
    print(f"💾 Checkpoint saved & verified: {generic} ({size_mb:.2f} MB)  | mirror: {stage}{tag}")

# ---------- Grow / Shrink params when max_len changes ----------

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

def shrink_params(params, old_len, new_len, bond_dim_mps, bond_dim_mera):
    assert new_len < old_len and new_len % 2 == 0
    def shrink_block(block):
        mps_old = block['mps']; isos_old = block['isos']
        return {'mps': mps_old[:new_len], 'isos': isos_old[:(new_len//2)]}
    new_encoder = shrink_block(params['encoder'])
    new_decoder = shrink_block(params['decoder'])
    old_td = (old_len//2) * bond_dim_mera
    new_td = (new_len//2) * bond_dim_mera
    proj_old = params['projection_matrix']
    proj_new = proj_old[:new_td]
    return {
        'encoder': new_encoder,
        'decoder': new_decoder,
        'projection_matrix': proj_new,
        'output_projection': params['output_projection'],
    }

# ===================== Dynamic staging from corpus =====================

def length_histogram(sequences: List[List[str]], max_len_limit: int) -> List[int]:
    counts = [0]*(max_len_limit+1)
    for s in sequences:
        L = len(s)
        if 1 <= L <= max_len_limit:
            counts[L] += 1
    return counts

def greedy_stage_endpoints(counts: List[int], max_len_limit: int, target_per_stage: int) -> List[int]:
    endpoints = []; acc = 0
    for L in range(1, max_len_limit+1):
        acc += counts[L]
        if acc >= target_per_stage and (L % 2 == 0):
            endpoints.append(L); acc = 0
    last_even = max_len_limit if (max_len_limit % 2 == 0) else (max_len_limit-1)
    if not endpoints or endpoints[-1] != last_even:
        if last_even > 0 and (not endpoints or endpoints[-1] < last_even):
            endpoints.append(last_even)
    cleaned = []; prev = 0
    for L in endpoints:
        if L > prev and L % 2 == 0:
            cleaned.append(L); prev = L
    return cleaned

def build_dynamic_stages_from_corpus(sequences: List[List[str]], max_len_limit: int,
                                     auto_max_stages: int, min_bucket_for_stage: int) -> List[int]:
    assert max_len_limit % 2 == 0
    counts = length_histogram(sequences, max_len_limit)
    total = sum(counts[1:])
    if total == 0:
        return [max_len_limit]
    tgt = max(min_bucket_for_stage, math.ceil(total / max(1, auto_max_stages)))
    for _ in range(10):
        endpoints = greedy_stage_endpoints(counts, max_len_limit, tgt)
        if len(endpoints) <= auto_max_stages:
            return endpoints
        tgt = int(tgt * 1.25)
    return greedy_stage_endpoints(counts, max_len_limit, tgt)

def make_stage_bounds(length_stages: List[int], start_min_len: int = 1) -> List[Tuple[int,int]]:
    bounds = []; prev = max(1, int(start_min_len) - 1)
    for L in length_stages:
        bounds.append((prev+1, L)); prev = L
    return bounds

def stage_stats(all_sequences: List[List[str]], bounds_list: List[Tuple[int,int]]) -> Tuple[List[int], List[int], List[float]]:
    counts = [0]*len(bounds_list); tok_sums = [0]*len(bounds_list)
    for s in all_sequences:
        L = len(s)
        for i, (low, high) in enumerate(bounds_list):
            if low <= L <= high:
                counts[i] += 1; tok_sums[i] += L; break
    means = [ (tok_sums[i]/counts[i]) if counts[i]>0 else 0.0 for i in range(len(bounds_list)) ]
    return counts, tok_sums, means

def compute_stage_fractions(bounds_list: List[Tuple[int,int]], counts: List[int], tok_sums: List[int], mode: str = "by_tokens") -> List[float]:
    if mode not in ("uniform", "by_tokens"):
        mode = "by_tokens"
    raw = [1.0 if c>0 else 0.0 for c in counts] if mode=="uniform" else [float(tok) for tok in tok_sums]
    s = sum(raw)
    if s <= 0:
        return [1.0/len(bounds_list)]*len(bounds_list)
    return [x/s for x in raw]

def compute_targets(num_samples: int, fracs: List[float], avail: List[int]) -> List[int]:
    assert len(fracs) == len(avail)
    total_avail = sum(avail)
    budget = min(int(num_samples), total_avail)
    if budget <= 0: return [0]*len(fracs)
    s = float(sum(fracs))
    fracs = [f/s for f in fracs] if s>0 else [1.0/len(fracs)]*len(fracs)
    raw = [budget * f for f in fracs]
    base = [min(int(x), a) for x,a in zip(raw, avail)]
    rema = [x - int(x) for x in raw]
    used = sum(base); left = budget - used
    order = sorted(range(len(fracs)), key=lambda i: rema[i], reverse=True)
    for i in order:
        if left<=0: break
        if base[i] < avail[i]:
            base[i]+=1; left-=1
    return base

def select_stage_sequences_no_upsample(all_sequences: List[List[str]], bounds: Tuple[int,int],
                                       target_count: int, seed: int) -> Tuple[List[List[str]], List[List[str]]]:
    low, high = bounds
    candidates = [s for s in all_sequences if low <= len(s) <= high]
    r = random.Random(seed); r.shuffle(candidates)
    k = min(target_count, len(candidates))
    selected = candidates[:k]; extra_pool = candidates[k:]
    return selected, extra_pool

# ===================== Hard-Example Mining helpers =====================

def compute_stage_batch_size(max_len: int) -> int:
    return max(MIN_STAGE_BATCH, min(BATCH_SIZE, max(1, TOKENS_PER_BATCH_TARGET // max(1, max_len))))

def compute_eval_batch_size(max_len: int, tpb: int) -> int:
    return max(1, min(BATCH_SIZE, max(1, tpb // max(1, max_len))))

def hard_mine_from_extra_pool(params, processor, extra_pool: List[List[str]],
                              max_len: int, pad_idx: int, curr_smoothing: float,
                              need: int, rng_seed: int) -> Tuple[List[List[str]], List[List[str]], Dict[str,int]]:
    if need <= 0 or len(extra_pool) == 0:
        return [], extra_pool, {'cands':0,'eval_batches':0,'picked':0}

    r = random.Random(rng_seed)
    n_cands = min(len(extra_pool), max(1, int(len(extra_pool) * HEM_CANDIDATE_FRAC)))
    n_cands = min(n_cands, HEM_CANDIDATE_MAX)
    cand_idx = r.sample(range(len(extra_pool)), n_cands)
    candidates = [extra_pool[i] for i in cand_idx]

    cand_arr = build_arrays(processor, candidates, max_len)
    eval_batch = compute_eval_batch_size(max_len, HEM_EVAL_TOKENS_PER_BATCH)
    losses = []
    rng = jax.random.PRNGKey(1337)
    for i in range(0, len(cand_arr), eval_batch):
        chunk = jnp.asarray(cand_arr[i:i+eval_batch])
        lvec = per_seq_losses(params, chunk, pad_idx, rng, 0.0, 0.0)
        losses.append(np.array(lvec))
    losses = np.concatenate(losses, axis=0) if losses else np.array([])

    k = min(need, len(candidates))
    top_idx_local = np.argsort(-losses)[:k]  # descending
    picked = [candidates[i] for i in top_idx_local]

    pick_global = set(cand_idx[i] for i in top_idx_local)
    new_extra = [s for i,s in enumerate(extra_pool) if i not in pick_global]

    stats = {'cands': n_cands, 'eval_batches': int(math.ceil(n_cands / eval_batch)), 'picked': len(picked)}
    return picked, new_extra, stats

def upsample_train_set_general(
    base_train: List[List[str]],
    extra_pool: List[List[str]],
    factor: float,
    repeat_cap_multiplier: float,
    seed: int,
    strategy: str,
    params=None, processor=None, max_len=None, pad_idx=None, curr_smoothing: float=0.0
) -> Tuple[List[List[str]], List[List[str]], Dict[str,int]]:
    r = random.Random(seed)
    n0 = len(base_train)
    if n0 == 0:
        return base_train, extra_pool, {'added_unique':0, 'added_repeats':0, 'target':0, 'final':0, 'strategy':strategy}

    max_allowed = int(max(1, n0 * repeat_cap_multiplier))
    desired = int(min(max_allowed, max(n0+1, n0 * factor)))

    new_train = list(base_train)
    new_extra = list(extra_pool)

    need = desired - len(new_train)
    added_unique = 0
    if need > 0 and len(new_extra) > 0:
        if strategy == 'hard' and ENABLE_HEM and HEM_ON_UPSAMPLE and params is not None:
            mined, new_extra, _ = hard_mine_from_extra_pool(
                params, processor, new_extra, max_len, pad_idx, curr_smoothing, need, seed+4242
            )
            new_train.extend(mined); added_unique += len(mined); need = desired - len(new_train)
        if need > 0 and len(new_extra) > 0:
            r.shuffle(new_extra)
            take = min(need, len(new_extra))
            new_train.extend(new_extra[:take]); new_extra = new_extra[take:]; added_unique += take

    need = desired - len(new_train)
    repeats_used = 0
    if need > 0:
        pool = list(new_train)
        for _ in range(need):
            new_train.append(r.choice(pool)); repeats_used += 1

    stats = {'added_unique': added_unique, 'added_repeats': repeats_used,
             'target': desired, 'final': len(new_train), 'strategy': strategy}
    return new_train, new_extra, stats

# ===================== Adaptive upsampling trigger =====================

def _rel_improve(old: float, new: float) -> float:
    if not (np.isfinite(old) and np.isfinite(new)) or old <= 0:
        return 0.0
    return max(0.0, (old - new) / old)

def should_adaptive_upsample(
    epoch_idx: int,
    val_hist: List[float],
    best_val: float,
    last_upsample_epoch: int,
    min_trigger_epoch: int,
    plateau_patience: int,
    window: int,
    min_rel_improve: float,
    abs_ceiling: Optional[float],
    cooldown_epochs: int,
    min_required: Optional[float] = None,   # continue nudging until <= min_required
) -> Tuple[bool, str]:
    """
    After an upsample, compute plateau/low_slope relative only to the history *since* that event.
    If min_required is set and we're still above it, demand a plateau (or abs_ceiling) to trigger,
    to avoid spamming back-to-back upsampling on mere low_slope signals.
    """
    e = epoch_idx + 1
    if e < min_trigger_epoch:
        return (False, "warmup")

    # Cooldown from the last upsample
    if last_upsample_epoch >= 0 and (e - last_upsample_epoch) < cooldown_epochs:
        return (False, "cooldown")

    anchor = max(0, last_upsample_epoch)
    hist_since = val_hist[anchor:] if anchor < len(val_hist) else []
    if len(hist_since) < 2:
        return (False, "insufficient_data")

    plateau = False
    if len(hist_since) >= plateau_patience + 1:
        tail = hist_since[-(plateau_patience+1):]
        improved = any(tail[i] + MIN_DELTA < tail[i-1] for i in range(1, len(tail)))
        plateau = not improved

    low_slope = False
    if len(hist_since) >= window + 1:
        before = hist_since[-(window+1)]
        now = hist_since[-1]
        rel = _rel_improve(before, now)
        low_slope = (rel < min_rel_improve)

    too_high = (abs_ceiling is not None and hist_since[-1] > abs_ceiling)
    above_min = (min_required is not None and hist_since[-1] > min_required)

    if above_min:
        trigger = plateau or too_high
        reason  = "plateau" if plateau else ("abs_ceiling" if too_high else "above_min_no_plateau")
    else:
        trigger = plateau or low_slope or too_high
        if plateau:
            reason = "plateau"
        elif low_slope:
            reason = "low_slope"
        elif too_high:
            reason = "abs_ceiling"
        else:
            reason = "none"

    return (trigger, reason)


# ===================== Schedules =====================

def make_optimizer(total_steps: int):
    warmup_steps = max(1, int(total_steps * WARMUP_FRAC))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=BASE_LR,
        warmup_steps=warmup_steps,
        decay_steps=max(1, total_steps - warmup_steps),
        end_value=BASE_LR * COSINE_FINAL_RATIO,
    )
    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=WEIGHT_DECAY)
    )
    return opt

# ===================== Train/Val/Test split =====================

def split_for_len(seqs, seed=42, ratios=(0.8,0.1,0.1)):
    r = random.Random(seed); seqs = list(seqs); r.shuffle(seqs)
    n=len(seqs); n_train=int(n*ratios[0]); n_val=int(n*ratios[1])
    return seqs[:n_train], seqs[n_train:n_train+n_val], seqs[n_train+n_val:]

# ===================== Multi-GPU helpers =====================

def shard_first_axis(x: np.ndarray, ndev: int):
    """Reshape [B, *] -> [ndev, per_dev, *]. Assumes B % ndev == 0."""
    B = x.shape[0]
    per = B // ndev
    new_shape = (ndev, per) + tuple(x.shape[1:])
    return x.reshape(new_shape)

def shard_pairs(per_device_pairs, per_device_masks):
    """Stack list of per-device (M,2)/(M,) into arrays [D,M,2] and [D,M]."""
    pairs = np.stack(per_device_pairs, axis=0)
    masks = np.stack(per_device_masks, axis=0)
    return pairs, masks

# ===================== Main =====================

def main():
    # Reproducibility
    random.seed(PY_SEED); np.random.seed(PY_SEED)

    # Devices
    N_DEV = jax.local_device_count()
    IS_MULTI = USE_PMAP_MULTI_GPU and (N_DEV > 1)
    print(f"🖥️  JAX devices: {N_DEV} | multi-GPU={'ON' if IS_MULTI else 'OFF'} (pmap)")

    # Checks
    assert FIXED_MAX_LEN_CAP % 2 == 0, "FIXED_MAX_LEN_CAP must be even"
    assert MAX_LEN_LIMIT % 2 == 0, "MAX_LEN_LIMIT must be even"
    assert 1 <= MIN_LEN_LIMIT <= MAX_LEN_LIMIT, "MIN_LEN_LIMIT must be within [1, MAX_LEN_LIMIT]"

    print("📦 Setup:")
    print(f"  MIN_LEN_LIMIT       = {MIN_LEN_LIMIT}")
    print(f"  MAX_LEN_LIMIT       = {MAX_LEN_LIMIT}")
    print(f"  NUM_SAMPLES         = {NUM_SAMPLES}")
    print(f"  AUTO_MAX_STAGES     = {AUTO_MAX_STAGES}")
    print(f"  MIN_BUCKET_FOR_STAGE= {MIN_BUCKET_FOR_STAGE}")
    print(f"  FRAC_MODE           = {FRAC_MODE}")
    print(f"  BATCH_SIZE(max)     = {BATCH_SIZE}")
    print(f"  TPB target          = {TOKENS_PER_BATCH_TARGET}")
    print(f"  Adaptive Upsampling = {ADAPTIVE_UPSAMPLING}, HEM={ENABLE_HEM}")
    print(f"  HEM candidates frac/max = {HEM_CANDIDATE_FRAC}/{HEM_CANDIDATE_MAX}")
    print(f"  MIN_VAL_PPL_REQUIRED= {MIN_VAL_PPL_REQUIRED}")
    print(f"  SAVE_LATEST_EVERY   = {SAVE_LATEST_EVERY} (0=off)")
    print(f"  SAVE_BEST_EVERY     = {SAVE_BEST_EVERY} (0=buffer-only, 1=immediate, N=periodic)")

    processor = DataProcessor(sequence_type='protein')
    sequences = processor.load_stockholm_ungapped(
        DATA_FILES, min_len=MIN_LEN_LIMIT, max_len_cap=FIXED_MAX_LEN_CAP
    )
    print(f"Full corpus size (deduped, len>={MIN_LEN_LIMIT}): {len(sequences)}")
    random.shuffle(sequences)

    processor.build_vocab()
    VOCAB_SIZE = processor.word_count
    print(f"Vocab size = {VOCAB_SIZE}")

    # ----- Dynamic stages -----
    LENGTH_STAGES = build_dynamic_stages_from_corpus(
        sequences,
        max_len_limit=MAX_LEN_LIMIT,
        auto_max_stages=AUTO_MAX_STAGES,
        min_bucket_for_stage=MIN_BUCKET_FOR_STAGE
    )
    print("Dynamic LENGTH_STAGES =", LENGTH_STAGES)
    
    # ➜ Force single stage όταν ζητάς 1 (ή 0) stages
    if AUTO_MAX_STAGES <= 1:
        last_even = MAX_LEN_LIMIT if (MAX_LEN_LIMIT % 2 == 0) else (MAX_LEN_LIMIT - 1)
        LENGTH_STAGES = [last_even]
    
    # Προαιρετικό ενδιάμεσο eval stage — OFF
    TARGET_EVAL_LEN = None
    if (TARGET_EVAL_LEN is not None
        and TARGET_EVAL_LEN % 2 == 0
        and MIN_LEN_LIMIT <= TARGET_EVAL_LEN <= MAX_LEN_LIMIT
        and TARGET_EVAL_LEN not in LENGTH_STAGES):
        LENGTH_STAGES = sorted(set(LENGTH_STAGES + [TARGET_EVAL_LEN]))
    
    print("Adjusted LENGTH_STAGES =", LENGTH_STAGES)


    bounds_list = make_stage_bounds(LENGTH_STAGES, start_min_len=MIN_LEN_LIMIT)
    counts, tok_sums, mean_lens = stage_stats(sequences, bounds_list)

    # Prune empty stages
    if any(c == 0 for c in counts):
        pruned = [LENGTH_STAGES[i] for i,c in enumerate(counts) if c == 0]
        print("ℹ️ Pruning empty stages:", pruned)
        LENGTH_STAGES = [LENGTH_STAGES[i] for i,c in enumerate(counts) if c > 0]
        bounds_list = make_stage_bounds(LENGTH_STAGES, start_min_len=MIN_LEN_LIMIT)
        counts, tok_sums, mean_lens = stage_stats(sequences, bounds_list)

    fracs = compute_stage_fractions(bounds_list, counts, tok_sums, mode=FRAC_MODE)
    per_stage_targets = compute_targets(NUM_SAMPLES, fracs, counts)

    # Build per-stage selected + extra_pool
    stages_data = []
    for i, (bounds, target) in enumerate(zip(bounds_list, per_stage_targets)):
        stage_seed = PY_SEED + 1000 + i
        selected, extra_pool = select_stage_sequences_no_upsample(sequences, bounds, target, stage_seed)
        stages_data.append({'bounds': bounds, 'selected': selected, 'extra_pool': extra_pool})
        print(f"Stage {i+1} [{bounds[0]},{bounds[1]}]: avail={counts[i]}, target={target}, selected={len(selected)}, extra_pool={len(extra_pool)}")

    total_selected = sum(len(s['selected']) for s in stages_data)
    if total_selected < NUM_SAMPLES:
        print(f"ℹ️ Selected {total_selected} < requested {NUM_SAMPLES} (no upsampling at allocation).")

    rng = RNG_KEY
    params = None
    opt_state = None
    pad_idx = processor.vocab['<PAD>']
    current_len = 0

    # ======= Resume (generic) =======
    start_stage_idx = 0
    generic_latest = os.path.join(CHECKPOINT_DIR, 'latest.pkl')
    resume_payload = try_load_checkpoint(generic_latest)
    if resume_payload is not None and 'params' in resume_payload:
        ckpt_len = resume_payload.get('max_len', resume_payload['params']['encoder']['mps'].shape[0])
        for i, L in enumerate(LENGTH_STAGES):
            if L >= ckpt_len:
                start_stage_idx = i; break
        print(f"🔁 Resume: found generic checkpoint (len={ckpt_len}). Will start from stage #{start_stage_idx+1} (MAX_LEN={LENGTH_STAGES[start_stage_idx]}).")
    else:
        print("🔁 Resume: no generic checkpoint found. Starting from the first stage.")

    # ------- helper to (re)build optimizer & step -------
    def build_opt_and_step(steps_est: int, helper_symbols: int):
        opt = make_optimizer(steps_est)
        if USE_SYMBOLICS:
            # ensure symbolic params before opt.init
            nonlocal params
            params = ensure_symbolic_params(params, helper_symbols, rng)
        state = opt.init(params)
        if IS_MULTI:
            step_fn = make_train_step_symbolic_pmap(opt, axis_name=PMAP_AXIS_NAME)
        else:
            step_fn = make_train_step_symbolic(opt) if USE_SYMBOLICS else None
        return opt, state, step_fn

    for stage_idx in range(start_stage_idx, len(LENGTH_STAGES)):
        MAX_LEN = LENGTH_STAGES[stage_idx]

        # Stage batch: if multi-GPU, force multiple of N_DEV
        base_stage_batch = compute_stage_batch_size(MAX_LEN)
        if IS_MULTI:
            per_dev = max(1, base_stage_batch // N_DEV)
            STAGE_BATCH = per_dev * N_DEV
        else:
            STAGE_BATCH = base_stage_batch

        print("\n======================================================================")
        print(f"   STAGE {stage_idx+1}/{len(LENGTH_STAGES)} — MAX_LEN = {MAX_LEN} | STAGE_BATCH = {STAGE_BATCH}{' ('+str(N_DEV)+'x'+str(STAGE_BATCH//N_DEV)+')' if IS_MULTI else ''}")
        print("======================================================================")

        stage_selected = stages_data[stage_idx]['selected']

        if len(stage_selected) == 0:
            print("⚠️  No sequences for this stage. Skipping.")
            continue

        # ------- Recover per-stage data_state if compatible -------
        data_state = None
        if resume_payload is not None:
            data_state = resume_payload.get('data_state', None)

        # default fresh splits
        use_recovered = False
        if (data_state is not None
            and data_state.get('stage_idx', -1) == stage_idx
            and tuple(data_state.get('bounds', (-1,-1))) == tuple(bounds_list[stage_idx])):
            # Restore exact splits & hyper-state
            train_seqs = data_state['train_seqs']
            val_seqs   = data_state['val_seqs']
            test_seqs  = data_state['test_seqs']
            did_upsample_events = int(data_state.get('did_upsample_events', 0))
            last_upsample_epoch = int(data_state.get('last_upsample_epoch', -1))
            curr_dropout        = float(data_state.get('curr_dropout', DROPOUT_RATE_TRAIN))
            curr_smoothing      = float(data_state.get('curr_smoothing', LABEL_SMOOTHING))
            # RNG states (best-effort)
            rng_pack = resume_payload.get('rng_state', {})
            try:
                if 'py' in rng_pack: random.setstate(rng_pack['py'])
                if 'np' in rng_pack: np.random.set_state(rng_pack['np'])
                if 'jax_key' in rng_pack: rng = rng_pack['jax_key']
            except Exception:
                pass
            use_recovered = True
            print("🔁 Recovered data_state for this stage (train/val/test + upsampling hyper-state).")
        else:
            train_seqs, val_seqs, test_seqs = split_for_len(stage_selected, seed=42)
            did_upsample_events = 0
            last_upsample_epoch = -1
            curr_dropout   = DROPOUT_RATE_TRAIN
            curr_smoothing = LABEL_SMOOTHING

        # Build arrays
        train_arr = build_arrays(processor, train_seqs, MAX_LEN)
        val_arr   = build_arrays(processor, val_seqs,   MAX_LEN)
        test_arr  = build_arrays(processor, test_seqs,  MAX_LEN)

        # Deterministic stage_extra excluding ALL unique in train/val/test
        used_unique = {seq_key(s) for s in (train_seqs + val_seqs + test_seqs)}
        stage_extra = rebuild_stage_extra(sequences, bounds_list[stage_idx], used_unique)

        # ---- SYMBOLICS for TRAIN ONLY (memory safe)
        if USE_SYMBOLICS:
            helper = SymbolicsHelper(processor, config=DEFAULT_PROTEIN_CONFIG)
            train_masks_u8, train_symv_u8, train_neighbors_local = helper.build_for_subset(train_seqs, MAX_LEN)
            print(f"Symbolics subset ready: train={len(train_seqs)}  masks={train_masks_u8.shape}")
        else:
            train_masks_u8 = np.ones((len(train_seqs), MAX_LEN-1, VOCAB_SIZE), dtype=np.uint8)
            train_symv_u8  = np.zeros((len(train_seqs), 1), dtype=np.uint8)
            train_neighbors_local = np.full((len(train_seqs), 1), -1, dtype=np.int32)

        steps_per_epoch = max(1, (len(train_arr) // STAGE_BATCH))
        total_steps_est = steps_per_epoch * EPOCHS_PER_STAGE

        # Load or init params (length-aware)
        loaded, loaded_path = load_checkpoint_for_len(CHECKPOINT_DIR, MAX_LEN)
        if loaded is not None:
            params, opt_state = loaded.get('params'), loaded.get('opt_state')
            start_epoch = loaded.get('epoch', -1) + 1
            ckpt_len = params['encoder']['mps'].shape[0]
            if ckpt_len < MAX_LEN:
                print(f"ℹ️  The checkpoint ({loaded_path}) has len={ckpt_len}. Growing to len={MAX_LEN} …")
                params = grow_params(params, ckpt_len, MAX_LEN, BOND_DIMENSION_MPS, BOND_DIMENSION_MERA, VOCAB_SIZE, rng)
                set_pad_identity(params, pad_idx)
                start_epoch = 0
            elif ckpt_len > MAX_LEN:
                print(f"ℹ️  The checkpoint ({loaded_path}) has len={ckpt_len}. Shrinking to len={MAX_LEN} …")
                params = shrink_params(params, ckpt_len, MAX_LEN, BOND_DIMENSION_MPS, BOND_DIMENSION_MERA)
                set_pad_identity(params, pad_idx)
                start_epoch = 0
            else:
                set_pad_identity(params, pad_idx)
            print(f"✅ Loaded checkpoint from '{loaded_path}' (epoch {start_epoch})")
        else:
            if params is not None and current_len > 0 and current_len != MAX_LEN:
                prev_len = current_len
                print(f"➡️  Growing parameters from {prev_len} ➜ {MAX_LEN} …")
                params = grow_params(params, prev_len, MAX_LEN, BOND_DIMENSION_MPS, BOND_DIMENSION_MERA, VOCAB_SIZE, rng)
                set_pad_identity(params, pad_idx)
                start_epoch = 0
            elif params is None:
                print("Starting fresh training… (init params)")
                params = init_encoder_decoder_model(RNG_KEY, VOCAB_SIZE, BOND_DIMENSION_MPS, BOND_DIMENSION_MERA, MAX_LEN)
                set_pad_identity(params, pad_idx)
                start_epoch = 0
            else:
                start_epoch = 0

        current_len = MAX_LEN

        # Build optimizer & step (single or multi)
        optimizer, opt_state, train_step_sym = build_opt_and_step(total_steps_est, helper.num_symbols if USE_SYMBOLICS else 0)

        # Warm-up JIT (single device suffices)
        if len(train_arr) >= STAGE_BATCH:
            dummy = jnp.asarray(train_arr[:min(STAGE_BATCH, len(train_arr))])
            rng, subk = jax.random.split(rng)
            _ = loss_fn(params, dummy, dummy, pad_idx, subk, 0.0, LABEL_SMOOTHING).block_until_ready()

        # ---- Training loop (adaptive upsampling + HEM) ----
        best_val = float('inf')
        best_epoch = -1
        best_payload_mem = None      # buffered best (in-memory)
        last_best_saved_epoch = -1   # for periodic best saving
        no_improve = 0
        stage_t0 = time.time()
        if not use_recovered:
            did_upsample_events = 0
            last_upsample_epoch = -1
            curr_dropout = DROPOUT_RATE_TRAIN
            curr_smoothing = LABEL_SMOOTHING

        val_hist: List[float] = []
        train_hist: List[float] = []
#####################WARMUP TRIGGER GIA UPSAMPLING
        min_trigger_epoch = max(1, int(0.1 * MIN_EPOCHS))

        for epoch in range(start_epoch, EPOCHS_PER_STAGE):
            t0=time.time()
            perm = np.random.permutation(len(train_arr))
            total_loss=0.0; nb=0; total_tokens=0

            # small schedules for symbolics
            alpha_e = 0.5 if USE_SYMBOLICS else 0.0
            tau_e   = 0.9 if USE_SYMBOLICS else 1.0
            lambda_graph_e = LAMBDA_GRAPH_BASE * (1.0 + 0.3 * min(1.0, epoch/50.0)) if USE_SYMBOLICS else 0.0

            for i in range(0, len(train_arr), STAGE_BATCH):
                idx = perm[i:i+STAGE_BATCH]
                if len(idx) < STAGE_BATCH:
                    continue

                if IS_MULTI:
                    # Shard inputs per device
                    batch_np = np.asarray(train_arr[idx])
                    masks_np = np.asarray(train_masks_u8[idx])
                    symv_np  = np.asarray(train_symv_u8[idx])

                    batch_sh = shard_first_axis(batch_np, N_DEV)
                    masks_sh = shard_first_axis(masks_np, N_DEV)
                    symv_sh  = shard_first_axis(symv_np,  N_DEV)

                    # Build pairs per device (host-side)
                    per_dev_pairs = []
                    per_dev_masks = []
                    per = STAGE_BATCH // N_DEV
                    for d in range(N_DEV):
                        local_idx = idx[d*per:(d+1)*per]
                        p_pad, p_mask = get_pairs_for_batch(
                            local_idx, train_neighbors_local,
                            edges_per_node=GRAPH_EDGES_PER_NODE,
                            max_pairs=GRAPH_MAX_PAIRS,
                        )
                        per_dev_pairs.append(np.array(p_pad))
                        per_dev_masks.append(np.array(p_mask))
                    pairs_sh, pmask_sh = shard_pairs(per_dev_pairs, per_dev_masks)

                    # RNG keys per device
                    rng, subk = jax.random.split(rng)
                    dev_keys = jax.random.split(subk, N_DEV)

                    params, opt_state, loss = train_step_sym(
                        params, opt_state,
                        jnp.asarray(batch_sh),
                        int(pad_idx),
                        dev_keys,
                        float(curr_dropout), float(curr_smoothing),
                        jnp.asarray(masks_sh),
                        jnp.asarray(symv_sh),
                        jnp.asarray(pairs_sh),
                        jnp.asarray(pmask_sh),
                        float(lambda_graph_e), float(alpha_e), float(tau_e),
                    )
                else:
                    batch = jnp.asarray(train_arr[idx])
                    rng, subk = jax.random.split(rng)

                    if USE_SYMBOLICS:
                        pairs_padded, pairs_mask = get_pairs_for_batch(
                            idx, train_neighbors_local,
                            edges_per_node=GRAPH_EDGES_PER_NODE,
                            max_pairs=GRAPH_MAX_PAIRS,
                        )
                        params, opt_state, loss = train_step_sym(
                            params, opt_state,
                            batch_inputs=batch,
                            pad_idx=pad_idx, rng_key=subk,
                            dropout_rate=curr_dropout, label_smoothing=curr_smoothing,
                            allowed_masks_uint8=jnp.asarray(train_masks_u8[idx]),
                            symbol_vecs_uint8=jnp.asarray(train_symv_u8[idx]),
                            pairs_padded=pairs_padded, pairs_mask=pairs_mask,
                            lambda_graph=lambda_graph_e, alpha=alpha_e, tau=tau_e,
                        )
                    else:
                        l, grads = jax.value_and_grad(loss_fn)(
                            params, batch, batch, pad_idx, subk, curr_dropout, curr_smoothing
                        )
                        updates, opt_state = optimizer.update(grads, opt_state, params)
                        params = optax.apply_updates(params, updates)
                        loss = l

                total_loss += float(loss); nb += 1
                total_tokens += int((np.asarray(train_arr[idx])[:,1:] != pad_idx).sum())

            train_ppl = float(np.exp(total_loss/nb)) if nb>0 else float('inf')
            val_ppl   = compute_perplexity(
                params, val_arr, pad_idx,
                batch=min(STAGE_BATCH, len(val_arr)),
                label_smoothing=0.0
            )
            dt = time.time()-t0
            print(f"[len={MAX_LEN:>3}, bs={STAGE_BATCH:>3}{' ('+str(N_DEV)+'x'+str(STAGE_BATCH//N_DEV)+')' if IS_MULTI else ''}] "
                  f"Epoch {epoch+1:>3}/{EPOCHS_PER_STAGE} | {dt:.1f}s | train-PPL {train_ppl:.3f} | val-PPL {val_ppl:.3f} | tokens {total_tokens}")

            train_hist.append(train_ppl); val_hist.append(val_ppl)

            # ---------- BEST (buffered / periodic) ----------
            improved = val_ppl + MIN_DELTA < best_val
            if improved:
                best_val = val_ppl
                best_epoch = epoch
                best_payload_mem = {
                    'params': params,
                    'opt_state': opt_state,
                    'epoch': epoch,
                    'max_len': current_len,
                    'data_state': {
                        'stage_idx': stage_idx,
                        'bounds': bounds_list[stage_idx],
                        'length_stages': LENGTH_STAGES,
                        'train_seqs': train_seqs,
                        'val_seqs':   val_seqs,
                        'test_seqs':  test_seqs,
                        'did_upsample_events': did_upsample_events,
                        'last_upsample_epoch': last_upsample_epoch,
                        'curr_dropout':   curr_dropout,
                        'curr_smoothing': curr_smoothing,
                    },
                    'rng_state': {
                        'py': random.getstate(),
                        'np': np.random.get_state(),
                        'jax_key': rng,
                    }
                }
                # Periodic write according to SAVE_BEST_EVERY
                should_write_best_now = (
                    SAVE_BEST and
                    ((SAVE_BEST_EVERY == 1) or (SAVE_BEST_EVERY > 1 and ((epoch + 1) % SAVE_BEST_EVERY == 0)))
                )
                if should_write_best_now:
                    try:
                        save_checkpoint_dual(best_payload_mem, CHECKPOINT_DIR, MAX_LEN, is_best=True)
                        last_best_saved_epoch = epoch + 1
                    except Exception as e:
                        print("❌ Failed to save best checkpoint (periodic):", e)
                else:
                    print(f"⭐ New best buffered (val-PPL {best_val:.3f}) — will flush later.")
                no_improve = 0
            else:
                no_improve += 1

            # ---------- PERIODIC 'LATEST' SAVE ----------
            if SAVE_LATEST_EVERY and SAVE_LATEST_EVERY > 0 and ((epoch + 1) % SAVE_LATEST_EVERY == 0):
                try:
                    payload = {
                        'params': params,
                        'opt_state': opt_state,
                        'epoch': epoch,
                        'max_len': current_len,
                        'data_state': {
                            'stage_idx': stage_idx,
                            'bounds': bounds_list[stage_idx],
                            'length_stages': LENGTH_STAGES,
                            'train_seqs': train_seqs,
                            'val_seqs':   val_seqs,
                            'test_seqs':  test_seqs,
                            'did_upsample_events': did_upsample_events,
                            'last_upsample_epoch': last_upsample_epoch,
                            'curr_dropout':   curr_dropout,
                            'curr_smoothing': curr_smoothing,
                        },
                        'rng_state': {
                            'py': random.getstate(),
                            'np': np.random.get_state(),
                            'jax_key': rng,
                        }
                    }
                    save_checkpoint_dual(payload, CHECKPOINT_DIR, MAX_LEN, is_best=False)
                except Exception as e:
                    print("❌ Failed to save periodic checkpoint:", e)

            # ---- Adaptive upsampling trigger ----
            try_upsample, reason = should_adaptive_upsample(
                epoch_idx=epoch, val_hist=val_hist, best_val=best_val,
                last_upsample_epoch=last_upsample_epoch,
                min_trigger_epoch=min_trigger_epoch,
                plateau_patience=UPSAMPLE_TRIGGER_PATIENCE,
                window=REL_IMPROVEMENT_WINDOW,
                min_rel_improve=MIN_REL_IMPROVEMENT,
                abs_ceiling=PPL_ABS_CEILING,
                cooldown_epochs=COOLDOWN_EPOCHS,
                min_required=MIN_VAL_PPL_REQUIRED,
            )

            if ADAPTIVE_UPSAMPLING and try_upsample and did_upsample_events < MAX_UPSAMPLE_EVENTS_PER_STAGE:
                old_n = len(train_seqs)
                desired_factor = min(MAX_UPSAMPLE_FACTOR, UPSAMPLE_FACTOR_BASE + UPSAMPLE_FACTOR_STEP * did_upsample_events)
                strategy = 'hard' if (ENABLE_HEM and HEM_ON_UPSAMPLE) else 'random'
                new_train_seqs, stage_extra, stats = upsample_train_set_general(
                    base_train=train_seqs,
                    extra_pool=stage_extra,
                    factor=desired_factor,
                    repeat_cap_multiplier=REPEAT_CAP_MULTIPLIER,
                    seed=PY_SEED + 777 + stage_idx*1000 + epoch,
                    strategy=strategy,
                    params=params, processor=processor, max_len=MAX_LEN, pad_idx=pad_idx,
                    curr_smoothing=curr_smoothing
                )
                if len(new_train_seqs) > old_n:
                    train_seqs = new_train_seqs
                    train_arr = build_arrays(processor, train_seqs, MAX_LEN)

                    # >>> REBUILD SYMBOLIC BUNDLES for NEW TRAIN SET <<<
                    if USE_SYMBOLICS:
                        helper = SymbolicsHelper(processor, config=DEFAULT_PROTEIN_CONFIG)
                        train_masks_u8, train_symv_u8, train_neighbors_local = helper.build_for_subset(train_seqs, MAX_LEN)
                    # <<< /REBUILD >>>

                    # Recompute steps & rebuild optimizer/step (reset opt state)
                    if IS_MULTI:
                        per_dev = max(1, compute_stage_batch_size(MAX_LEN) // N_DEV)
                        STAGE_BATCH = per_dev * N_DEV
                    else:
                        STAGE_BATCH = compute_stage_batch_size(MAX_LEN)

                    steps_per_epoch = max(1, (len(train_arr) // STAGE_BATCH))
                    total_steps_est = steps_per_epoch * max(1, (EPOCHS_PER_STAGE - epoch - 1))
                    optimizer, opt_state, train_step_sym = build_opt_and_step(total_steps_est, helper.num_symbols if USE_SYMBOLICS else 0)

                    curr_dropout = UPSAMPLED_DROPOUT
                    curr_smoothing = UPSAMPLED_SMOOTH
                    did_upsample_events += 1
                    last_upsample_epoch = epoch + 1
                    print(f"🟩 Upsample[{strategy}] {reason} at epoch {epoch+1}: "
                          f"train {old_n} → {len(train_seqs)} "
                          f"(+uniq {stats['added_unique']}, +rep {stats['added_repeats']}), "
                          f"dropout→{curr_dropout}, smoothing→{curr_smoothing}")
                    no_improve = 0
                else:
                    print(f"ℹ️ Upsample skipped (no capacity). Reason={reason}")

            # Early stop per stage (respect MIN_VAL_PPL_REQUIRED)
            allow_stop = (MIN_VAL_PPL_REQUIRED is None) or (best_val <= MIN_VAL_PPL_REQUIRED)
            if (epoch + 1) >= MIN_EPOCHS and no_improve >= PATIENCE:
                if allow_stop:
                    print(f"🟨 Plateau at len={MAX_LEN} (no improvement {PATIENCE}x after {MIN_EPOCHS}+ epochs, best val-PPL {best_val:.3f}). "
                          f"Stopping stage (threshold ok: {MIN_VAL_PPL_REQUIRED}).")
                    # Flush buffered BEST (if any)
                    if FLUSH_BEST_ON_EARLY_STOP and best_payload_mem is not None and SAVE_BEST:
                        try:
                            save_checkpoint_dual(best_payload_mem, CHECKPOINT_DIR, MAX_LEN, is_best=True)
                            last_best_saved_epoch = epoch + 1
                        except Exception as e:
                            print("❌ Failed to save BEST on early stop:", e)
                    # Force-save LATEST on early stop
                    if SAVE_LATEST_ON_EARLY_STOP:
                        try:
                            payload = {
                                'params': params,
                                'opt_state': opt_state,
                                'epoch': epoch,
                                'max_len': current_len,
                                'data_state': {
                                    'stage_idx': stage_idx,
                                    'bounds': bounds_list[stage_idx],
                                    'length_stages': LENGTH_STAGES,
                                    'train_seqs': train_seqs,
                                    'val_seqs':   val_seqs,
                                    'test_seqs':  test_seqs,
                                    'did_upsample_events': did_upsample_events,
                                    'last_upsample_epoch': last_upsample_epoch,
                                    'curr_dropout':   curr_dropout,
                                    'curr_smoothing': curr_smoothing,
                                },
                                'rng_state': {
                                    'py': random.getstate(),
                                    'np': np.random.get_state(),
                                    'jax_key': rng,
                                }
                            }
                            save_checkpoint_dual(payload, CHECKPOINT_DIR, MAX_LEN, is_best=False)
                        except Exception as e:
                            print("❌ Failed to save checkpoint on early stop:", e)
                    break
                else:
                    print(f"🟪 Early-stop deferred: best val-PPL {best_val:.3f} > required {MIN_VAL_PPL_REQUIRED:.3f}. "
                          f"Continuing training & upsampling as needed.")
                    no_improve = 0  # avoid spamming

        stage_dt = time.time()-stage_t0
        print(f"⏱️  Stage time (len={MAX_LEN}): {stage_dt:.1f}s")

        # Final saves at stage end
        # Flush buffered BEST (if any)
        if FLUSH_BEST_ON_STAGE_END and best_payload_mem is not None and SAVE_BEST:
            try:
                save_checkpoint_dual(best_payload_mem, CHECKPOINT_DIR, MAX_LEN, is_best=True)
                last_best_saved_epoch = best_payload_mem['epoch'] + 1
            except Exception as e:
                print("❌ Failed to save BEST at stage end:", e)

        # Force-save LATEST at stage end
        if SAVE_LATEST_ON_STAGE_END:
            try:
                payload = {
                    'params': params,
                    'opt_state': opt_state,
                    'epoch': epoch,  # last epoch run
                    'max_len': current_len,
                    'data_state': {
                        'stage_idx': stage_idx,
                        'bounds': bounds_list[stage_idx],
                        'length_stages': LENGTH_STAGES,
                        'train_seqs': train_seqs,
                        'val_seqs':   val_seqs,
                        'test_seqs':  test_seqs,
                        'did_upsample_events': did_upsample_events,
                        'last_upsample_epoch': last_upsample_epoch,
                        'curr_dropout':   curr_dropout,
                        'curr_smoothing': curr_smoothing,
                    },
                    'rng_state': {
                        'py': random.getstate(),
                        'np': np.random.get_state(),
                        'jax_key': rng,
                    }
                }
                save_checkpoint_dual(payload, CHECKPOINT_DIR, MAX_LEN, is_best=False)
            except Exception as e:
                print("❌ Failed to save checkpoint at stage end:", e)

        # Stage Test (use in-memory best if present)
        if best_payload_mem is not None:
            params = best_payload_mem['params']
        test_ppl = compute_perplexity(
            params, test_arr, pad_idx,
            batch=min(STAGE_BATCH, len(test_arr)),
            label_smoothing=0.0
        )
        print(f"✅ [len={MAX_LEN}] TEST Perplexity: {test_ppl:.3f}")

    # ---- SAVE FINAL MODEL ----
    try:
        atomic_pickle_save((params, processor, current_len), MODEL_FILE)
        size_mb = os.path.getsize(MODEL_FILE)/1024/1024
        print(f"\n✅ Saved: {MODEL_FILE} ({size_mb:.2f} MB)")
    except Exception as e:
        print("❌ Failed to save model:", e)

    total_dt = time.time()-start_global
    print(f"🏁 Curriculum complete. Total time: {total_dt:.1f}s")

# ===================== Entry =====================

if __name__ == '__main__':
    start_global = time.time()
    main()
