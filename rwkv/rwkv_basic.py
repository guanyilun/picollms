import jax.numpy as np
from jax.lax import rsqrt
from jax import jit
from jax.nn import sigmoid, relu

def layer_norm(x, weight, bias, eps: float = 1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return weight * (x - mean) * rsqrt(variance + eps) + bias

def time_mix(x, x_prev, mix):
    return mix * x + (1 - mix) * x_prev

def exp_mix_frac(p1, p2, v1_upper, v1_lower, v2_upper, v2_lower):
    p = np.maximum(p1, p2)
    e1 = np.exp(p1 - p)
    e2 = np.exp(p2 - p)
    return v1_upper * e1 + v2_upper * e2, v1_lower * e1 + v2_lower * e2, p

def rkv(x, x_prev, time_mix_r, time_mix_k, time_mix_v, r_proj, k_proj, v_proj):
    x_r = time_mix(x, x_prev, time_mix_r)
    x_k = time_mix(x, x_prev, time_mix_k)
    x_v = time_mix(x, x_prev, time_mix_v)
    r = sigmoid(r_proj @ x_r)
    k = k_proj @ x_k
    v = v_proj @ x_v
    return r, k, v

def token_mixing(x, x_prev, a_prev, b_prev, p_prev, time_mix_r, time_mix_k, time_mix_v, r_proj, k_proj, v_proj, o_proj, time_first, time_decay):
    u, w = time_first, time_decay
    r, k, v = rkv(x, x_prev, time_mix_r, time_mix_k, time_mix_v, r_proj, k_proj, v_proj)
    expkv, expk, p = v, np.ones_like(v), k
    a_state, b_state, p_state = exp_mix_frac(p_prev + w, p, a_prev, b_prev, expkv, expk)
    c, d, _ = exp_mix_frac(p_prev, p+u, a_prev, b_prev, expkv, expk)  # u+w is an approx to log(exp(u+w)-1)
    # alternative approximation: u+w \approx log(exp(u+w)-1)
    # c, d, _ = exp_mix_frac(p_state, p+u+w, a_state, b_state, expkv, expk)
    rwkv = r * (c / d)
    return o_proj @ rwkv, a_state, b_state, p_state

def channel_mixing(x, x_prev, time_mix_r, time_mix_k, r_proj, k_proj, v_proj):
    x_r = time_mix(x, x_prev, time_mix_r)
    x_k = time_mix(x, x_prev, time_mix_k)
    r = sigmoid(r_proj @ x_r)
    k = np.square(relu(k_proj @ x_k))
    return r * (v_proj @ k)

@jit
def rwkv_net(token, state, emb, blocks, ln_out, head):
    x = emb['weight'][token]

    # initial layer norm is stored in block 0 in default weight file
    ln0 = blocks[0]['ln0']
    x = layer_norm(x, **ln0)

    for i in range(len(blocks)):
        x_tm = layer_norm(x, **blocks[i]['ln1'])
        x_p, a_state, b_state, p_state = token_mixing(x_tm, state[i, 1], state[i, 2], state[i, 3], state[i, 4], **blocks[i]['att'])
        x += x_p

        x_cm = layer_norm(x, **blocks[i]['ln2'])
        x += channel_mixing(x_cm, state[i, 0], **blocks[i]['ffn'])

        state = state.at[i, 0].set(x_cm)
        state = state.at[i, 1].set(x_tm)
        state = state.at[i, 2].set(a_state)
        state = state.at[i, 3].set(b_state)
        state = state.at[i, 4].set(p_state)

    x = layer_norm(x, **ln_out)
    logits = head['weight'] @ x

    return logits, state
