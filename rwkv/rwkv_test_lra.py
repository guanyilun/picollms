import jax
from jax import numpy as np
from functools import partial

from rwkv_batch import rwkv_net_batch
from lra_utils import LRABatchConfig, lra_acc_fn

# initialize LRA dataset
cache_path = "lra_benchmarks"
lra_config = LRABatchConfig.from_s5(32, cache_path, "listops-classification")

weights = np.load('wandb/latest-run/files/rwkv_weights.npy', allow_pickle=True).item()
acc_fn = jax.jit(partial(lra_acc_fn, rwkv_net_batch))

def get_test_results(test_dataloader, weights):
    n_batch = 0
    acc = []
    for batch in test_dataloader:
        n_batch += 1
        acc.append(acc_fn(weights, batch))
    print(f"Test accuracy:", np.mean(np.array(acc)))

get_test_results(lra_config.get_dataloader('test'), weights)