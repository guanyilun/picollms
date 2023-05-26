"""Similar to rwkv_basic but used for training so it doesn't
keep track of any states"""

from jax import vmap, jit
import jax.numpy as np
from jax import lax
from einops import repeat, einsum
import rwkv_basic as basic
from functools import partial

def rkv(x, time_mix_r, time_mix_k, time_mix_v, r_proj, k_proj, v_proj):
    # x: (n_seq, n_embed)
    x_prev  = np.concatenate([np.zeros_like(x[:1, ...]), x[:-1, ...]], axis=0)
    rkv_p   = vmap(basic.rkv, in_axes=(0, 0, None, None, None, None, None, None), out_axes=(0,0,0))
    r, k, v = rkv_p(x, x_prev, time_mix_r, time_mix_k, time_mix_v, r_proj, k_proj, v_proj)
    return r, k, v

def assoc_reduce_step(left, right, w):
    (expkv_l, expk_l, p_l) = left
    (expkv_r, expk_r, p_r) = right
    a, b, p = basic.exp_mix_frac(p_l + w, p_r, expkv_l, expk_l, expkv_r, expk_r)
    out = (a, b, p)
    return out, out

def token_mixing(x, time_mix_r, time_mix_k, time_mix_v, r_proj, k_proj, v_proj, o_proj, time_decay, time_first):
    # x: (n_seq, n_embed)
    u, w = time_first, time_decay
    r, k, v = rkv(x, time_mix_r, time_mix_k, time_mix_v, r_proj, k_proj, v_proj)
    expkv, expk, p = v, np.ones_like(v), k
    state = (expkv[0]*0, expk[0]*0, p[0]*0)
    step_f = partial(assoc_reduce_step, w=w)
    (a_state, b_state, p_state), (_, _, _) = lax.scan(step_f, state, (expkv, expk, p))
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
        x += token_mixing(x_tm, **blocks[i]['att'])
        x_cm = basic.layer_norm(x, **blocks[i]['ln2'])
        x += channel_mixing(x_cm, **blocks[i]['ffn'])
    x = basic.layer_norm(x, **ln_out)
    # head: {'weight': (n_vocab, n_embed)}
    logits = einsum(x, head['weight'], 's e, v e -> s v')
    return logits