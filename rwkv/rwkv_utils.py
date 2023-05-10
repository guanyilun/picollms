import jax
from jax import numpy as np

def get_tokenizer():
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file("20B_tokenizer.json")
    return tokenizer

def parse_rwkv_weight(pth_path="RWKV-4-Pile-169M-20220807-8023.pth"):
    import torch
    raw_weights = torch.load(pth_path, map_location='cpu')
    w = {}
    for key in raw_weights.keys():
        parts = key.split('.')
        last = parts.pop()
        current_ = w
        for p in parts:
            if p.isdigit():
                p = int(p)
            if p not in current_:
                current_[p] = {}
            current_ = current_[p]
        val = raw_weights[key]
        if '.time_' in key:
            val = val.squeeze()
        if '.time_decay' in key:
            val = -torch.exp(val)
        current_[last] = val.float().numpy()

    for i in w['blocks'].keys():
        att = w['blocks'][i]['att']
        ffn = w['blocks'][i]['ffn']

        for m in att, ffn:
            for k in ('key', 'value', 'receptance', 'output'):
                if k in m:
                    m[k[0] + '_proj'] = m[k]['weight']
                    del m[k]
    return w

def parse_model_info(weights_tree):
    return {
        'n_layer': len(weights_tree['blocks']),
        'n_embed': weights_tree['emb']['weight'].shape[1],
        'n_vocab': weights_tree['emb']['weight'].shape[0],
    }

def rnn_generate(model, weights_tree, prompt, n_tokens=50, tokenizer=None, state=None):
    model_info = parse_model_info(weights_tree)
    if state is None:
        state = np.zeros((model_info['n_layer'], 5, model_info['n_embed']))
    if tokenizer is None:
        tokenizer = get_tokenizer()
    input_ids = tokenizer.encode(prompt).ids
    print(prompt)
    for input_id in input_ids[:-1]:
        _, state = model(input_id, state, **weights_tree)
    input_id = input_ids[-1]
    res = prompt + ": "
    for _ in range(n_tokens):
        logits, state = model(input_ids[-1], state, **weights_tree)
        out_id = np.argmax(logits)
        out_token = tokenizer.decode([out_id])
        print(out_token, end="")
        res += out_token
        input_ids.append(out_id)
    return res

def rnn_generate_batch_stateless(model, weights_tree, prompt, n_tokens=10, tokenizer=None):
    if tokenizer is None:
        tokenizer = get_tokenizer()
    input_ids = tokenizer.encode(prompt).ids
    print(prompt, end='')
    for _ in range(n_tokens):
        input_ids_batch = np.array(input_ids).reshape(1, -1)
        out = model(input_ids_batch, **weights_tree)
        out_id = np.argmax(out[0, -1, :])
        res = tokenizer.decode([out_id])
        print(res, end='')
        input_ids.pop(0)  # jit becomes very slow when shape changes
        input_ids.append(out_id)
    print()
