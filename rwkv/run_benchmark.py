#%%
import jax
import jax.numpy as jnp
import numpy as np
import optax

import rwkv_utils as utils
from benchmark.rwkv_parallel_scan import rwkv_net as rwkv_net_parallel_scan
from benchmark.rwkv_parallel_scan_folded import rwkv_net as rwkv_net_parallel_scan_folded
from benchmark.rwkv_serial_scan import rwkv_net as rwkv_net_serial_scan
import time

tokenizer = utils.get_tokenizer()
weights = utils.parse_rwkv_weight("pretrain/RWKV-4-PilePlus-169M-20230505-3102-512Gtokens-ctx4096.pth")
# not sure why I have to do this device put myself, my guess is that torch loads the weights as
# pinned memory in CPU and jax doesn't like that
weights = jax.tree_map(lambda x: jax.device_put(x.astype(jax.numpy.bfloat16)), weights)

# model_f = rwkv_net_parallel_scan
# ofile_fwd = "benchmark/parallel_scan_fixed_fwd.txt"
# ofile_bwd = "benchmark/parallel_scan_fixed_bwd.txt"
# model_f = rwkv_net_serial_scan
# ofile_fwd = "benchmark/serial_scan_fixed_fwd.txt"
# ofile_bwd = "benchmark/serial_scan_fixed_bwd.txt"
model_f = rwkv_net_parallel_scan_folded
# ofile_fwd = "benchmark/parallel_scan_folded_fwd_256.txt"
# ofile_bwd = "benchmark/parallel_scan_folded_bwd_256.txt"
ofile_fwd = "benchmark/parallel_scan_folded_fwd_1024.txt"
ofile_bwd = "benchmark/parallel_scan_folded_bwd_1024.txt"

# n_tokens = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]#, 50000]#, 100000]#, 1000000]
# n_tokens = [20000]#, 50000]#, 100000]#, 1000000]
n_tokens = np.arange(1, 8) * 1024
# n_tokens = [2048]#, 10000, 20000, 50000]#, 100000]#, 1000000]
n_tokens = list(reversed(n_tokens))

#%% forward
print("Benchmarking forward pass")
timings = []
for n_token in n_tokens:
    print("Benchmarking with {} tokens".format(n_token))
    input_ids = jax.device_put(jnp.array(tokenizer.encode("dummy"*n_token), dtype=jnp.int32))
    # warm up
    print("warming up...")
    for i in range(2):
        model_f(input_ids, **weights)
    batch_timings = []

    # with jax.profiler.trace('/tmp/jaxtrace', True):
    for i in range(20):
        t0 = time.time()
        print(f"actually running timing test {i}...")
        model_f(input_ids, **weights)
        t1 = time.time()
        dt = t1 - t0
        batch_timings.append(dt)
    timings.append(np.median(batch_timings))
    print(timings)
np.savetxt(ofile_fwd, timings)

#%% backward

print("Benchmarking backward pass")
def get_loss_fn(model_f):
    def loss_fn(weights, x):
        logits = model_f(x, **weights)
        return optax.softmax_cross_entropy_with_integer_labels(logits, x).mean() # dummy to test backward
    return loss_fn
loss_fn = jax.jit(jax.grad(get_loss_fn(model_f)))

timings = []
for n_token in n_tokens:
    print("Benchmarking with {} tokens".format(n_token))
    input_ids = jax.device_put(jnp.array(tokenizer.encode("dummy"*n_token), dtype=jnp.int32))
    # warm up
    for i in range(2):
        print("warming up...")
        loss_fn(weights, input_ids)
    batch_timings = []
    for i in range(20):
        print(f"actually running timing test {i}...")
        t0 = time.time()
        loss_fn(weights, input_ids)
        t1 = time.time()
        dt = t1 - t0
        batch_timings.append(dt)
    timings.append(np.median(batch_timings))
np.savetxt(ofile_bwd, timings)