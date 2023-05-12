"""train rwkv using long-range arena benchmark dataset"""
from absl import logging

import jax
from jax import jit, numpy as np
from jax.nn.initializers import zeros, glorot_normal
import optax
import os.path as op

from rwkv_batch import rwkv_net_batch
from rwkv_utils import get_tokenizer, rnn_generate_batch_stateless, parse_rwkv_weight
import rwkv_train_utils as tu
from data_utils import load_dataset

use_wandb = True
adam_params = {
    'learning_rate': 1e-4,
    'b1': 0.9,
    'b2': 0.999,
    'eps': 1e-8,
}
lion_params = {
    'learning_rate': 1e-4,
    'b1': 0.95,
    'b2': 0.98,
    'weight_decay': 0.01
}
run_config = {
    'name': 'rwkv',
    'dataset': 'english',
    'n_epoch': 100,
    'batch_size': 8,
    'eval_freq': 200,
    'n_channel': 512,
    'n_layer': 8,
    'n_ffn': 1024,
    # 'opt': 'adam',
    # 'opt_params': adam_params,
    'opt': 'lion',
    'opt_params': lion_params,
    'block_size': 256,
    'save_step': 5000,
    'n_kernel': 50,
}

if use_wandb:
    import wandb
    wandb_run = wandb.init(
        project="inside-transformer",
        config=run_config,
    )

tokenizer = get_tokenizer()

# initialize weights
keygen = tu.KeyGen()
logging.info("initializing weights...")
winfo = tu.init_weight_info(
    tokenizer.get_vocab_size(),
    run_config['n_channel'],
    run_config['n_layer'],
    run_config['n_ffn'],
    run_config['n_kernel'],
)
# option 1:
# all zero init but head and embedding
# weights = init_weights(winfo, keygen, zeros)  # key is not required for zeros init
# weights['emb']['weight'] = init_uniform(keygen(), winfo['emb']['weight'], a=-1e-4, b=1e-4)
# weights['head']['weight'] = init_uniform(keygen(), winfo['head']['weight'], a=-1e-4, b=1e-4)

# option 2:
# glorot_normal for all 2d matrices and zero for all 1d vectors
# w_init_fn = lambda key, shape: glorot_normal()(key, shape) if len(shape) == 2 else zeros(key, shape)
# weights = init_weights(winfo, keygen, w_init_fn)

# option 3:
# load existing weights as starting point
# weights = parse_rwkv_weight("pretrain/RWKV-4-Pile-169M-20220807-8023.pth")
# logging.info("weights initialized")

# option 4:
# ref_weights = parse_rwkv_weight("pretrain/RWKV-4-Pile-169M-20220807-8023.pth")
ref_weights = np.load("rwkv_weights_10000.npy", allow_pickle=True).item()
logging.info("weights initialized")
# weights = tu.init_weights_by_resampling_matching_tree(winfo, keygen, ref_weights)
# weights = tu.init_weights_by_resampling_with_rule(winfo, keygen, ref_weights, tu.match_rule)
weights = tu.init_weights_by_resampling_with_rule(winfo, keygen, ref_weights, tu.match_rule_conv)

# initialize optimizers
logging.info("initializing optimizer...")
optimizer = {'lion': optax.lion, 'adam': optax.adam}[run_config['opt']](**run_config['opt_params'])
opt_state = optimizer.init(weights)
logging.info("optimizer initialized")

# setup loss, its grad, accuracy and validation
loss_fn = tu.get_loss_fn(rwkv_net_batch)
loss_fn_grad = jax.value_and_grad(loss_fn)
acc_fn = tu.get_acc_fn(rwkv_net_batch)

def get_validation_results(weights):
    prompt = "Hamlet once said that to be or not to be, "
    output = rnn_generate_batch_stateless(rwkv_net_batch, weights, prompt, 50, tokenizer)
    res = {'output': output}
    return res

@jit
def make_step(weights, opt_state, batch):
    loss_val, grads = loss_fn_grad(weights, batch)
    updates, opt_state = optimizer.update(grads, opt_state, weights)
    weights = optax.apply_updates(weights, updates)
    return weights, opt_state, loss_val

i_step = 0
logging.info("start training...")
for _ in range(run_config['n_epoch']):
    trainloader = load_dataset(dataset=run_config['dataset'], batch_size=run_config['batch_size'], block_size=run_config['block_size'])
    for batch in trainloader:
        weights, opt_state, loss_val = make_step(weights, opt_state, batch)
        if i_step % 10 == 0:
            print(f"step: {i_step}, batch loss: {loss_val}")
        if i_step % run_config['eval_freq'] == 0:
            res = get_validation_results(weights)
            if use_wandb:
                wandb.log({
                    "batch_loss": loss_val,
                    "n_tokens_trained": i_step * run_config['batch_size'] * run_config['block_size'],
                    # "generated": wandb.Html(res['output'])
                })
        if "n_train_step" in run_config and i_step >= run_config['n_train_step']:
            break
        if i_step % run_config['save_step'] == 0:
            ofile = f"rwkv_weights_{i_step}.npy"
            np.save(ofile, weights)
        i_step += 1

# ofile = op.join(wandb_run.dir, "rwkv_weights.npy") if use_wandb else "rwkv_weights.npy"
ofile = "rwkv_weights.npy"
np.save(ofile, weights)

if use_wandb: wandb.finish()

# example loading saved weights with np
# res = np.load("rwkv_weights.npy", allow_pickle=True).item()
