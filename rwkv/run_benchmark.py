#%%
import jax
jax.config.update('jax_platform_name', 'cpu')

import rwkv_utils as utils
from benchmark.rwkv_serial_scan import rwkv_net as rwkv_net_serial_scan
import time

# todo list:
# 1. DONE load pretrained model
# 2. DONE load tokenizer
# 3. DONE load input text
# 4. DONE run model with input text
# 5. add timers

#%%
tokenizer = utils.get_tokenizer()
weights = utils.parse_rwkv_weight("pretrain/RWKV-4-Pile-169M-20220807-8023.pth")

#%%
n_token = 50
input_text = "dummy"*50
input_ids = tokenizer.encode(input_text)

#%%
# warm up (jax need to trace and compile the function)
n_tokens = [10, 100, 1000, 10000, 100000, 1000000]
n_tokens = reversed(n_tokens)
timings = []
for n_token in n_tokens:
    print("Benchmarking with {} tokens".format(n_token))
    input_ids = tokenizer.encode("dummy"*n_token)
    # warm up
    print("warming up...")
    for i in range(2):
        rwkv_net_serial_scan(input_ids, **weights)
    t0 = time.time()
    rwkv_net_serial_scan(input_ids, **weights)
    t1 = time.time()
    dt = t1 - t0
    timings.append(dt)
# %%
