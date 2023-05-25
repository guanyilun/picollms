import jax
import numpy as np
# jax.config.update('jax_platform_name', 'cpu')

import rwkv_utils as utils
from benchmark.rwkv_parallel_scan import rwkv_net as rwkv_net_parallel_scan
from benchmark.rwkv_serial_scan import rwkv_net as rwkv_net_serial_scan
import time

tokenizer = utils.get_tokenizer()
weights = utils.parse_rwkv_weight("pretrain/RWKV-4-PilePlus-169M-20230505-3102-512Gtokens-ctx4096.pth")
weights = jax.tree_map(lambda x: x.astype(jax.numpy.bfloat16), weights)

n_tokens = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]#, 100000]#, 1000000]
n_tokens = reversed(n_tokens)
timings = []
for n_token in n_tokens:
    print("Benchmarking with {} tokens".format(n_token))
    input_ids = tokenizer.encode("dummy"*n_token)
    # warm up
    print("warming up...")
    for i in range(2):
        # rwkv_net_parallel_scan(input_ids, **weights)
        rwkv_net_serial_scan(input_ids, **weights)
    batch_timings = []
    for i in range(5):
        t0 = time.time()
        # rwkv_net_parallel_scan(input_ids, **weights)
        rwkv_net_serial_scan(input_ids, **weights)
        t1 = time.time()
        dt = t1 - t0
        batch_timings.append(dt)
    timings.append(np.median(batch_timings))
np.savetxt("benchmark/serial_scan_debug.txt", timings)