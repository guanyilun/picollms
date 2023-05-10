from rwkv_basic import rwkv_net
from rwkv_batch import rwkv_net_batch
from rwkv_utils import get_tokenizer, parse_rwkv_weight, parse_model_info, rnn_generate, rnn_generate_batch_stateless

tokenizer = get_tokenizer()
weights_tree = parse_rwkv_weight()
model_info = parse_model_info(weights_tree)

#%%
prompt = "The quick brown fox jumps over the lazy"

# test serial version
rnn_generate(rwkv_net, weights_tree, prompt, n_tokens=50, tokenizer=tokenizer)

# test batch version
# rnn_generate_batch_stateless(rwkv_net_batch, weights_tree, prompt, n_tokens=100, tokenizer=tokenizer)
