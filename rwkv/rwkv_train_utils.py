import jax
from jax import numpy as np
from jax.nn.initializers import uniform
import optax
import jax.tree_util as jtu
import fnmatch

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
                'time_mix_a': (n_channel,),
                'time_mix_b': (n_channel,),
                'time_decay': (n_channel,),
                'time_first': (n_channel,),
            },
            'ffn': {
                'k_proj': (n_ffn, n_channel),
                'v_proj': (n_channel, n_ffn),
                'r_proj': (n_channel, n_channel),
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

def resample_weights(key, target_shape, ref_weights: np.ndarray):
    """Resample weights to given shape"""
    flat_weights = ref_weights.flatten()
    target_size = np.prod(np.array(target_shape))
    random_indices = jax.random.randint(key, (target_size,), 0, flat_weights.shape[0])
    resampled_weights = flat_weights[random_indices]
    resampled_weights = np.reshape(resampled_weights, target_shape)
    return resampled_weights

def init_weights_by_resampling_matching_tree(weight_info, keygen, reference_wtree):
    """Resample weights from reference_wtree to match the shape of weight_info. It requires
    weight_info to have the same tree structure as reference_wtree, but the values
    can have different shapes. This means that for rwkv, as long as the n_layer matches, one
    is free to vary embedding size, n_ffn, etc."""
    return jax.tree_map(lambda x, y: resample_weights(keygen(), x, y), weight_info, reference_wtree, is_leaf=lambda x: isinstance(x, tuple))

def fold_winfo(winfo):
    """Fold nested weight info into a flat dict. The keys are dot-separated paths to the weights.
    The values are the shape of the weights.

    Example:
    { 'emb': {'weight': (n_vocab, n_channel)},
      'blocks': {0: {'att': {'o_proj': (n_channel, n_channel)}}}}
    becomes
    { 'emb.weight': (n_vocab, n_channel),
      'blocks.0.att.o_proj': (n_channel, n_channel)}

    Also returns the tree definition `re` which can be used to unfold the flat dict back to the nested structure.
    """
    flat_winfo, re = jax.tree_flatten(
        jax.tree_util.tree_map_with_path(lambda p, x: (".".join([str(p_.key) for p_ in p]), x), winfo, is_leaf=lambda x: isinstance(x, tuple)), 
        is_leaf=lambda x: isinstance(x, tuple)
    )
    return {k: v for (k, v) in flat_winfo}, re

def fold_wtree(wtree):
    """similar to fold_winfo, but for weights instead of weight info"""
    flat_weights, re = jtu.tree_flatten(
        jtu.tree_map_with_path(lambda p, x: (".".join([str(p_.key) for p_ in p]), x), wtree, is_leaf=lambda x: isinstance(x, np.ndarray)),
        is_leaf=lambda x: isinstance(x, tuple)
    )
    return {k: v for (k, v) in flat_weights}, re

# match_rule works like this:
# if match_rule.key in flat_winfo.key:
#    1. find all keys in reference weights that match the patterns
#       specified in match_rule.value
#    2. collect the weights in reference weight tree with matching
#       patterns and stack them together after flattening
#    3. resample a specific number of weights from the flattened
#       reference weights to form the new weights
match_rule = {
    "emb"       : "*emb*",
    "head"      : "*head*",
    "att.k_proj": "*att.k_proj",
    "att.v_proj": "*att.v_proj",
    "att.o_proj": "*att.o_proj",
    "att.r_proj": "*att.r_proj",
    "time_decay": "*time_decay",
    "time_first": "*time_first",
    "time_mix_a": "*att.time_mix_v",
    "time_mix_b": "*att.time_mix_r",
    "ffn.k_proj": "*ffn.k_proj",
    "ffn.v_proj": "*ffn.v_proj",
    "ffn.r_proj": "*ffn.r_proj",
    "ln0.weight": "*ln0.weight",
    "ln0.bias"  : "*ln0.bias",
    "ln1.weight": "*ln1.weight",
    "ln1.bias"  : "*ln1.bias",
    "ln2.weight": "*ln2.weight",
    "ln2.bias"  : "*ln2.bias",
    "ln_out.weight" : "*ln_out.weight",
    "ln_out.bias"   : "*ln_out.bias",
}

def init_weights_by_resampling_with_rule(winfo, keygen, ref_weights, match_rule=match_rule):
    # get flat version of both winfo and ref_weights
    # maintain the tree structure for reconstruction later
    flat_winfo, re = fold_winfo(winfo)
    flat_ref_weights, _ = fold_wtree(ref_weights)

    ref_weights_keys = list(flat_ref_weights.keys())
    flat_weights = {}
    # for each key in target weights, find a matching rule
    # and collect all matching weights, flatten, stack and resample
    for k, shape in flat_winfo.items():
        for (rule, query) in match_rule.items():
            found_match = False
            if rule in k:
                matched_keys = fnmatch.filter(ref_weights_keys, query)
                if len(matched_keys) == 0:
                    raise ValueError(f"no matching key found for {k} in match_rule")
                # collect all matching weights, flatten them and stack them
                print(f"Initializing {k} with: {matched_keys}")
                flat_ = np.hstack([flat_ref_weights[w].flatten() for w in matched_keys])
                flat_weights[k] = resample_weights(keygen(), shape, flat_)
                found_match = True
                # allow only one match
                break
        # if no match found, raise error.
        # TODO: add a default rule to match all weights
        if not found_match:
            raise ValueError(f"key {k} not matched in match_rule")
    # reconstruct the tree structure
    return jtu.tree_unflatten(re, list(flat_weights.values()))


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
