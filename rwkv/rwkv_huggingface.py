#%%
import os
from transformers import PretrainedConfig
from transformers import PreTrainedModel

import jax
import jax.numpy as np
from jax.nn.initializers import zeros

from rwkv_train_utils import init_weight_info, init_weights, init_uniform
from rwkv_batch import rwkv_net_batch


class RWKVConfig(PretrainedConfig):
    model_type = "rwkv"

    def __init__(
        self,
        n_layer: int = 4,
        n_classes_in: int = 20,
        n_classes_out: int = 20,
        n_channel: int = 512,
        n_ffn: int = 1024,
        jax_seed: int = 0,
        weights_name: str = "rwkv_weights",
        **kwargs,
    ):
        self.n_layer = n_layer
        self.n_classes_in = n_classes_in
        self.n_classes_out = n_classes_out
        self.n_channel =  n_channel
        self.n_ffn = n_ffn
        self.jax_seed = jax_seed
        self.weights_name = weights_name
        super().__init__(**kwargs)


class RWKV(PreTrainedModel):
    config_class = RWKVConfig

    def __init__(self, config):
        super().__init__(config)
        winfo = init_weight_info(
            config.n_classes_in,
            config.n_channel,
            config.n_layer,
            config.n_ffn,
            config.n_classes_out
        )
        key = jax.random.PRNGKey(config.jax_seed)
        weights = init_weights(winfo, None, zeros) # key is not required for zeros init
        weights['head']['weight'] = init_uniform(key, winfo['head']['weight'], a=-1e-4, b=1e-4)
        self.weights = weights
        self.weights_name = config.weights_name

    def forward(self, x):
        return rwkv_net_batch(x, **self.weights)

    def save_pretrained(self, save_directory):
        if os.path.isfile(save_directory):
            print("Provided path ({}) should be a directory, not a file".format(save_directory))
            return
        os.makedirs(save_directory, exist_ok=True)
        # get abs dir
        save_directory = os.path.abspath(save_directory)
        # save config as well
        self.config.save_pretrained(save_directory)
        # save model
        oname = os.path.join(save_directory, self.weights_name)
        np.save(oname, self.weights)

    @classmethod
    def from_pretrained(cls, pretrained_model_path):
        # get abs dir
        pretrained_model_path = os.path.abspath(pretrained_model_path)
        # pretrained_model_path is the directory
        config = RWKVConfig.from_pretrained(pretrained_model_path)
        ifile = os.path.join(pretrained_model_path, config.weights_name) + ".npy"
        weights = np.load(ifile, allow_pickle=True).item()
        obj = cls(config)
        obj.weights = weights
        return obj

RWKVConfig.register_for_auto_class()
RWKV.register_for_auto_class("AutoModel")

# %%
# debug
# rwkv_config = RWKVConfig()
# rwkv = RWKV(rwkv_config)
# rwkv.save_pretrained("test_hfsave")
# rwkv = RWKV.from_pretrained("pretrained/rwkv-4-pile-169m")
# rwkv(np.ones((1, 20), dtype=np.int32))

# %%
# convert pth to npy
# from rwkv_utils import parse_rwkv_weight
# weights = parse_rwkv_weight()
# np.save("pretrained/rwkv-4-pile-169m/rwkv-4-pile-169m.npy", weights)
