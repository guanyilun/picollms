"""Similar to rwkv_basic but used for training so it doesn't
keep track of any states"""

from jax import vmap, jit
import jax.numpy as np
from jax import lax
from einops import einsum, rearrange, repeat
import rwkv_basic as basic
from functools import partial

def rkv(x, time_mix_r, time_mix_k, time_mix_v, r_proj, k_proj, v_proj):
    # x: (n_seq, n_embed)
    x_prev  = np.concatenate([np.zeros_like(x[:1, ...]), x[:-1, ...]], axis=0)
    rkv_p = vmap(basic.rkv, in_axes=(0, 0, None, None, None, None, None, None), out_axes=(0,0,0))
    r, k, v = rkv_p(x, x_prev, time_mix_r, time_mix_k, time_mix_v, r_proj, k_proj, v_proj)
    return r, k, v

def assoc_reduce_step(left, right, w):
    (expkv_l, expk_l, n_l, p_l) = left
    (expkv_r, expk_r, n_r, p_r) = right
    a, b, p = basic.exp_mix_frac(p_l + n_r * w, p_r, expkv_l, expk_l, expkv_r, expk_r)
    return a, b, n_l + n_r, p

def batch_assoc_reduce_step(carry, new, w):
    (a_state, b_state, p_state) = carry           # (1, n_embed)
    (a_new, b_new, p_new) = new                   # (n_seq, n_embed)
    n = np.arange(a_new.shape[0]).reshape(-1, 1)  # (n_seq, 1)
    rescale = n * w.reshape(1, -1)                # (n_seq, n_embed)
    a_new, b_new, p_new = basic.exp_mix_frac(p_state + rescale, p_new, a_state, b_state, a_new, b_new)
    carry = (a_new[-1,:].reshape(1, -1), b_new[-1,:].reshape(1, -1), p_new[-1,:].reshape(1, -1))
    new = (a_new, b_new, p_new)
    return carry, new

def token_mixing(x, n_fold, time_mix_r, time_mix_k, time_mix_v, r_proj, k_proj, v_proj, o_proj, time_decay, time_first):
    # x: (n_seq, n_embed)
    u, w = time_first, time_decay
    r, k, v = rkv(x, time_mix_r, time_mix_k, time_mix_v, r_proj, k_proj, v_proj)
    x_ = rearrange(x, '(b s) e -> s (b e)', s=n_fold)
    k_ = rearrange(x, '(b s) e -> s (b e)', s=n_fold)
    v_ = rearrange(x, '(b s) e -> s (b e)', s=n_fold)
    n_ = np.ones_like(x, shape=(n_fold, 1))
    w_ = repeat(w, 'e -> (b e)', b=x.shape[0]//n_fold)
    expkv_, expk_, p_ = v_, np.ones_like(v_), k_

    # seq reduce
    reduce_step = partial(assoc_reduce_step, w=w_)
    a_state_, b_state_, _, p_state_ = lax.associative_scan(reduce_step, (expkv_, expk_, n_, p_))
    a_state = rearrange(a_state_, 's (b e) -> b s e', e=u.shape[0])
    b_state = rearrange(b_state_, 's (b e) -> b s e', e=u.shape[0])
    p_state = rearrange(p_state_, 's (b e) -> b s e', e=u.shape[0])

    # batch reduce
    batch_reduce_step = partial(batch_assoc_reduce_step, w=w)
    init = (
        np.zeros_like(a_state, shape=(1, a_state.shape[-1])), 
        np.zeros_like(b_state, shape=(1, b_state.shape[-1])), 
        np.zeros_like(p_state, shape=(1, p_state.shape[-1]))
    )
    state, (a_state_, b_state_, p_state_) = lax.scan(batch_reduce_step, init, (a_state, b_state, p_state))
    a_state = rearrange(a_state_, 'b s e -> (b s) e')
    b_state = rearrange(b_state_, 'b s e -> (b s) e')
    p_state = rearrange(p_state_, 'b s e -> (b s) e')

    # produce output y = c/d
    expkv   = rearrange(expkv_,   's (b e) -> (b s) e', e=u.shape[0])
    expk    = rearrange(expk_,    's (b e) -> (b s) e', e=u.shape[0])
    p       = rearrange(p_,       's (b e) -> (b s) e', e=u.shape[0])
    c, d, _ = basic.exp_mix_frac(p_state, p + u + w, a_state, b_state, expkv, expk)
    rwkv = c / d
    return (r * rwkv) @ o_proj.T

def channel_mixing(x, time_mix_r, time_mix_k, r_proj, k_proj, v_proj):
    # x: (n_seq, n_embed)
    x_prev  = np.concatenate([np.zeros_like(x[:1, ...]), x[:-1, ...]], axis=0)
    channel_mixing_p = vmap(basic.channel_mixing, in_axes=(0, 0, None, None, None, None, None), out_axes=0)
    out = channel_mixing_p(x, x_prev, time_mix_r, time_mix_k, r_proj, k_proj, v_proj)
    return out

@jit
def rwkv_net(token, emb, blocks, ln_out, head):
    # token: (n_seq,)
    x = emb['weight'][token, :]  # (n_seq, n_embed)
    ln0 = blocks[0]['ln0']
    x = basic.layer_norm(x, **ln0)
    for i in range(len(blocks)):
        x_tm = basic.layer_norm(x, **blocks[i]['ln1'])
        x += token_mixing(x_tm, n_fold=512, **blocks[i]['att'])
        x_cm = basic.layer_norm(x, **blocks[i]['ln2'])
        x += channel_mixing(x_cm, **blocks[i]['ffn'])
    x = basic.layer_norm(x, **ln_out)
    # head: {'weight': (n_vocab, n_embed)}
    logits = einsum(x, head['weight'], 's e, v e -> s v')
    return logits