import yaml
import random
import numpy as np
import torch



def set_seed(seed):
    # set seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed+10000)
    random.seed(seed)



def set_config(args):
    # load data config
    with open("./config/data_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        data_config = config[args.data_config]
    # load model config
    model_config = {}
    model_config["conditioner_name"] = args.conditioner
    model_config["transformer_name"] = args.transformer
    model_config["base_dist_name"] = args.base_dist
    with open("./config/model_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        model_config["normflow"] = config["normflow"]
        model_config["conditioner"] = config[args.conditioner+"_conditioner"]
        model_config["transformer"] = config[args.transformer+"_transformer"]
        model_config["base_dist"] = config[args.base_dist+"_base_dist"]
    
    return data_config, model_config