from jax import numpy as np
import jax
from jax.nn.initializers import uniform
import optax

# ==========================
# Loss / Acc related
# ==========================

def get_loss_fn(model_f):
    def loss_fn(weights, batch):
        x, y, _ = batch
        logits = model_f(x, **weights)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    return loss_fn

def get_acc_fn(model_f):
    def acc_fn(weights, batch):
        x, y, _ = batch
        logits = model_f(x, **weights)
        return (logits.argmax(axis=-1) == y).mean()
    return acc_fn

# ==========================
# Initialization related
# ==========================

# every call returns a new key
class KeyGen:
    def __init__(self, seed=0):
        self.key = jax.random.PRNGKey(seed)
    def __call__(self):
        self.key, new_key = jax.random.split(self.key)
        return new_key

def init_weight_info(n_vocab, n_channel, n_layer, n_ffn, n_vocab_out=None):
    # default to the same vocab size for output
    n_vocab_out = n_vocab_out or n_vocab
    info = {
        'emb': {'weight': (n_vocab, n_channel)},
        'blocks': {},
        'ln_out': {'weight': (n_channel,), 'bias': (n_channel,)},
        'head': {'weight': (n_vocab_out, n_channel)},
    }
    for i in range(n_layer):
        block = {
            'att': {
                'o_proj': (n_channel, n_channel),
                'k_proj': (n_channel, n_channel),
                'v_proj': (n_channel, n_channel),
                'r_proj': (n_channel, n_channel),
                'time_mix_r': (n_channel,),
                'time_mix_k': (n_channel,),
                'time_mix_v': (n_channel,),
                'time_decay': (n_channel,),
                'time_first': (n_channel,),
            },
            'ffn': {
                'k_proj': (n_ffn, n_channel),
                'v_proj': (n_channel, n_ffn),
                'r_proj': (n_channel, n_channel),
                'time_mix_k': (n_channel,),
                'time_mix_r': (n_channel,),
            },
            'ln1': {'weight': (n_channel,), 'bias': (n_channel,)},
            'ln2': {'weight': (n_channel,), 'bias': (n_channel,)},
        }
        # convention in rwkv: ln0 is in first block
        if i == 0: block['ln0'] = {'weight': (n_channel,), 'bias': (n_channel,)}
        info['blocks'][i] = block
    return info

def init_weights(weight_info, keygen, init_fn, **kwargs):
    return jax.tree_map(lambda x: init_fn(keygen(), x, **kwargs), weight_info, is_leaf=lambda x: isinstance(x, tuple))

def init_uniform(key, shape, a=-1e-4, b=1e-4, dtype=np.float32):
    # uniform in [a, b) range, default to [-1e-4, 1e-4) following rwkv recommendation
    return uniform(scale=b-a)(key, shape, dtype=dtype) + a

# ==========================
# Filename Utilities
# taken from:
#   https://github.com/cwhy/rwkv-decon/blob/main/python_utils.py
# ==========================

def format_num(num, unit):
    num_str = "{:.1f}".format(num).replace(".", "_")
    return num_str.rstrip("0").rstrip("_") + unit

def num_short_form(num):
    if num == 0:
        return "0"
    abs_num = abs(num)
    sign = "-" if num < 0 else ""
    if abs_num < 1000:
        return str(num)
    elif abs_num < 1000000:
        return sign + format_num(abs_num / 1000, "K")
    elif abs_num < 1000000000:
        return sign + format_num(abs_num / 1000000, "M")
    else:
        return sign + format_num(abs_num / 1000000000, "B")
