"""Similar to rwkv_basic but used for training so it doesn't
keep track of any states"""

import jax
from jax import vmap, jit
import jax.numpy as np
from jax.nn import softmax
from jax import lax
from einops import rearrange, repeat, einsum
from rwkv_basic import rkv, exp_mix_frac, channel_mixing, layer_norm

def time_conv(x, kernel):
    # x: (1, n_seq, n_batch x n_embed) -> CWN
    # kernel: (1, 1, kernel_size,) -> IOW
    # output: (1, n_seq, n_batch x n_embed) -> CWN
    kernel = softmax(kernel, axis=-1)
    kernel_size = kernel.shape[-1]
    dn = lax.conv_dimension_numbers(x.shape, kernel.shape, ('CWN', 'IOW', 'CWN'))
    return lax.conv_general_dilated(x, kernel,
                                    (1,), # window strides
                                    # [(kernel_size, -1)], # padding mode: 1 step dilated causal convolution up to x_{t-1}
                                    [(kernel_size-1, 0)], # padding mode: 0 step dilated causal convolution up to x_{t}
                                    (1,), # lhs/image dilation
                                    (1,), # rhs/kernel dilation
                                    dn)

def rkv_batch(x, time_kernel_r, time_kernel_v, r_proj, k_proj, v_proj):
    # x: (n_seq, n_batch, n_embed)
    x_ = rearrange(x, 's b e -> s (b e)').reshape((1, x.shape[0], -1))
    time_kernel_r_ = time_kernel_r.reshape((1,1,-1))
    time_kernel_v_ = time_kernel_v.reshape((1,1,-1))

    # convolve over time
    x_r_ = time_conv(x_, time_kernel_r_).reshape((x.shape[0], -1))
    x_k_ = x_.reshape((x.shape[0], -1))  # no mixing for k
    x_v_ = time_conv(x_, time_kernel_v_).reshape((x.shape[0], -1))

    x_r = rearrange(x_r_, 's (b e) -> (s b) e', b=x.shape[1])
    x_k = rearrange(x_k_, 's (b e) -> (s b) e', b=x.shape[1])
    x_v = rearrange(x_v_, 's (b e) -> (s b) e', b=x.shape[1])
    
    rkv_p = vmap(rkv, in_axes=(0, 0, 0, None, None, None), out_axes=(0,0,0))
    r_, k_, v_ = rkv_p(x_r, x_k, x_v, r_proj, k_proj, v_proj)

    r = rearrange(r_, '(s b) e -> s b e', s=x.shape[0])
    k = rearrange(k_, '(s b) e -> s b e', s=x.shape[0])
    v = rearrange(v_, '(s b) e -> s b e', s=x.shape[0])

    return r, k, v

def assoc_reduce_step(left, right):
    (expkv_l, expk_l, w_l, p_l) = left
    (expkv_r, expk_r, w_r, p_r) = right
    a, b, p = exp_mix_frac(p_l + w_r, p_r, expkv_l, expk_l, expkv_r, expk_r)
    return a, b, w_l + w_r, p

def token_mixing_batch(x, time_kernel_r, time_kernel_v, r_proj, k_proj, v_proj, o_proj, time_decay, time_first):
    """All this annoying rearranging is to try to do as much work for each
    reduction step so the overhead from multiprocessing becomes negligible. It may
    have very little effect, but it's worth a try."""
    u, w = time_first, time_decay
    r, k, v = rkv_batch(x, time_kernel_r, time_kernel_v, r_proj, k_proj, v_proj)
    k_ = rearrange(k, 's b e -> s (b e)')
    v_ = rearrange(v, 's b e -> s (b e)')
    W_ = repeat(time_decay, 'e -> s (b e)', s=r.shape[0], b=r.shape[1])
    expkv_, expk_, p_ = v_, np.ones_like(v_), k_
    a_state_, b_state_, _, p_state_ = lax.associative_scan(assoc_reduce_step, (expkv_, expk_, W_, p_))
    a_state = rearrange(a_state_, 's (b e) -> s b e', b=r.shape[1])
    b_state = rearrange(b_state_, 's (b e) -> s b e', b=r.shape[1])
    p_state = rearrange(p_state_, 's (b e) -> s b e', b=r.shape[1])
    expkv   = rearrange(expkv_, 's (b e) -> s b e', b=r.shape[1])
    expk    = rearrange(expk_, 's (b e) -> s b e', b=r.shape[1])
    p       = rearrange(p_, 's (b e) -> s b e', b=r.shape[1])
    c, d, _ = exp_mix_frac(p_state, p + u + w, a_state, b_state, expkv, expk)
    rwkv = c / d
    return (r * rwkv) @ o_proj.T

def channel_mixing_batch(x, r_proj, k_proj, v_proj):
    # x: (n_seq, n_batch, n_embed)
    x_ = rearrange(x, 's b e -> (s b) e')
    channel_mixing_p = vmap(channel_mixing, in_axes=(0, None, None, None), out_axes=0)
    out_ = channel_mixing_p(x_, r_proj, k_proj, v_proj)
    return rearrange(out_, '(s b) e -> s b e', s=x.shape[0])

@jit
def rwkv_net_batch(token, emb, blocks, ln_out, head):
    # token: (n_batch, n_seq) -> (n_seq, n_batch) for performance
    token = rearrange(token, 'b s -> s b')
    x = emb['weight'][token, :]
    # initial layer norm is stored in block 0 in default weight file
    ln0 = blocks[0]['ln0']
    x = layer_norm(x, **ln0)
    for i in range(len(blocks)):
        x_tm = layer_norm(x, **blocks[i]['ln1'])
        x += token_mixing_batch(x_tm, **blocks[i]['att'])
        x_cm = layer_norm(x, **blocks[i]['ln2'])
        x += channel_mixing_batch(x_cm, **blocks[i]['ffn'])
    x = layer_norm(x, **ln_out)
    # head: {'weight': (n_vocab, n_embed)}
    # put vocab back in last dimension for performance
    logits = einsum(x, head['weight'], 's b e, v e -> b s v')
    return logits
