"""train rwkv using long-range arena benchmark dataset"""
import jax
from jax import jit, numpy as np
# from jax.nn.initializers import zeros, glorot_normal

import optax
import wandb
import os.path as op
from functools import partial

from rwkv_batch import rwkv_net_batch
import rwkv_train_utils as tu
# from rwkv_utils import parse_rwkv_weight
from lra_utils import LRABatchConfig, lra_loss_fn, lra_acc_fn

use_wandb = True
adam_params = {
    'learning_rate': 1e-4,
    'b1': 0.9,
    'b2': 0.999,
    'eps': 1e-8,
}
adamw_params = {
    'learning_rate': 1e-3,
    'b1': 0.9,
    'b2': 0.999,
    'eps': 1e-8,
    'weight_decay': 0.01
}
lion_params = {
    'learning_rate': 1e-4,
    'b1': 0.95,
    'b2': 0.98,
    'weight_decay': 0.01
}
run_config = {
    'name': 'rwkv-lra',
    'n_epoch': 1000,
    'batch_size': 64,
    'eval_freq': 200,
    # 'n_train_step': 5000, # or n_epoch, whichever comes first
    # 'n_train_step': 5000*4, # or n_epoch, whichever comes first
    # 'n_channel': 512,
    'n_channel': 96,
    'n_layer': 4,
    'n_ffn': 192,
    # 'opt': 'adam',
    # 'opt_params': adam_params,
    # 'opt': 'lion',
    # 'opt_params': lion_params,
    'opt': 'adamw',
    'opt_params': adamw_params,
    'block_size': 2048,  # S5 default
    'n_kernel': 5,
}

if use_wandb:
    wandb_run = wandb.init(
        project="lra-listops",
        config=run_config,
    )

# initialize LRA dataset
cache_path = "lra_benchmarks"
lra_config = LRABatchConfig.from_s5(run_config['batch_size'], cache_path, "listops-classification")

# initialize weights
winfo = tu.init_weight_info(
    lra_config.n_classes_in,
    run_config['n_channel'],
    run_config['n_layer'],
    run_config['n_ffn'],
    run_config['n_kernel'],
    n_vocab_out=lra_config.n_classes_out
)

keygen = tu.KeyGen()
# option 1:
# all zero init but head and embedding
# weights = init_weights(winfo, None, zeros)  # key is not required for zeros init
# weights['emb']['weight'] = init_uniform(keygen(), winfo['emb']['weight'], a=-1e-4, b=1e-4)
# weights['head']['weight'] = init_uniform(keygen(), winfo['head']['weight'], a=-1e-4, b=1e-4)
# option 2:
# glorot_normal for all 2d matrices and zero for all 1d vectors
# w_init_fn = lambda key, shape: glorot_normal()(key, shape) if len(shape) == 2 else zeros(key, shape)
# weights = tu.init_weights(winfo, keygen, w_init_fn)
# option 3:
# ref_weights = parse_rwkv_weight("pretrain/RWKV-4-Pile-169M-20220807-8023.pth")
ref_weights = np.load("rwkv_weights_20000.npy", allow_pickle=True).item()
weights = tu.init_weights_by_resampling_with_rule(winfo, keygen, ref_weights, tu.match_rule_conv)

# initialize optimizers
optimizer = {'lion': optax.lion, 'adam': optax.adam, 'adamw': optax.adamw}[run_config['opt']](**run_config['opt_params'])
opt_state = optimizer.init(weights)

# setup loss, its grad, accuracy and validation
# loss_fn = partial(lra_loss_fn, rwkv_net_batch)
# loss_fn = partial(lra_loss_fn_all_tokens, rwkv_net_batch)
loss_fn = jax.jit(partial(lra_loss_fn, rwkv_net_batch))
loss_fn_grad = jax.value_and_grad(loss_fn)  # jitted in make_step
acc_fn = jax.jit(partial(lra_acc_fn, rwkv_net_batch))

def get_validation_results(val_dataloader, weights):
    val_loss = 0
    n_batch = 0
    acc = []
    for batch in val_dataloader:
        # not elegant: called model twice
        val_loss += loss_fn(weights, batch)
        n_batch += 1
        acc.append(acc_fn(weights, batch))
    res = {
        'validation_loss': val_loss / n_batch,
        'validation_acc': np.mean(np.array(acc)),
    }
    return res

@jit
def make_step(weights, opt_state, batch):
    loss_val, grads = loss_fn_grad(weights, batch)
    updates, opt_state = optimizer.update(grads, opt_state, weights)
    weights = optax.apply_updates(weights, updates)
    return weights, opt_state, loss_val

i_step = 0
done = False
for _ in range(run_config['n_epoch']):
    trainloader = lra_config.get_dataloader('train')
    for batch in trainloader:
        weights, opt_state, loss_val = make_step(weights, opt_state, batch)
        if i_step % run_config['eval_freq'] == 0:
            print(f"step: {i_step}, batch loss: {loss_val}")
            res = get_validation_results(lra_config.get_dataloader('val'), weights)
            if use_wandb:
                wandb.log({
                    "batch_loss": loss_val,
                    "validation_loss": res['validation_loss'],
                    "validation_acc": res['validation_acc'],
                    "n_tokens_trained": i_step * run_config['batch_size'] * run_config['block_size'],
                })
        if "n_train_step" in run_config and i_step >= run_config['n_train_step']:
            done = True
            break
        i_step += 1
    if done: break
ofile = op.join(wandb_run.dir, "rwkv_weights.npy") if use_wandb else "rwkv_weights.npy"
np.save(ofile, weights)

if use_wandb: wandb.finish()

# example loading saved weights with np
# res = np.load("rwkv_weights.npy", allow_pickle=True).item()
