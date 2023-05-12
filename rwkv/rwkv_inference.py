#%%
%load_ext autoreload
%autoreload 2
from rwkv_basic import rwkv_net
from rwkv_batch import rwkv_net_batch
from rwkv_utils import get_tokenizer, parse_rwkv_weight, parse_model_info, rnn_generate, rnn_generate_batch_stateless
import rwkv_train_utils as tu
from jax.nn.initializers import zeros, glorot_normal

tokenizer = get_tokenizer()
# weights_tree = parse_rwkv_weight()
# model_info = parse_model_info(weights_tree)
# initialize weights
keygen = tu.KeyGen()
winfo = tu.init_weight_info(
    tokenizer.get_vocab_size(),
    512,
    4,
    1024,
    100
)

#%%
# ref_weights = parse_rwkv_weight("pretrain/RWKV-4-Pile-169M-20220807-8023.pth")
ref_weights = np.load("rwkv_weights_10000.npy", allow_pickle=True).item()
# weights = tu.init_weights_by_resampling_matching_tree(winfo, keygen, ref_weights)
weights = tu.init_weights_by_resampling_with_rule(winfo, keygen, ref_weights, match_rule=tu.match_rule_conv)

#%%
prompt = "The quick brown fox jumps over the lazy"

# test serial version
# rnn_generate(rwkv_net, weights, prompt, n_tokens=50, tokenizer=tokenizer)

# test batch version
rnn_generate_batch_stateless(rwkv_net_batch, weights, prompt, n_tokens=100, tokenizer=tokenizer)

# %%
