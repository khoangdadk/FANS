import os
import torch



def set_fld(args, data_config):
    fld_data = f"d={data_config["d"]}_e={data_config["e"]}_N={data_config["n_samples"]}_dag={data_config["dag_type"]}_sem={data_config["sem_type"]}"
    fld_model = "nf" + "_" + args.conditioner + "_" + (args.transformer if args.conditioner != "bnaf" else "") + "_" + args.base_dist 
    return fld_data, fld_model



def save_model(model, best_model, fld_data, fld_model):
    if not os.path.exists(f'./reprod/{fld_data}/{fld_model}'):
        os.makedirs(f'./reprod/{fld_data}/{fld_model}')
    torch.save(model.state_dict(), f'./reprod/{fld_data}/{fld_model}/model.pt')
    torch.save(best_model.state_dict(), f'./reprod/{fld_data}/{fld_model}/bestVal_model.pt')



def load_model(model, fld_data, fld_model, fl):
    model.load_state_dict(torch.load(f'./reprod/{fld_data}/{fld_model}/{fl}.pt', weights_only=True))
    return model

