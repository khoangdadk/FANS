import math
import random
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from model.normflow import AMFlow
from model.energy import ACEModel
from data.masking import create_mask
from utils.graph import identify_mb
from metrics import ndcg_at_k



def run(args, 
        data_gg, 
        model, 
        device):
    
    if args.train:
        data = data_gg['data']['data']
        train_loader = DataLoader(data, batch_size=args.batch_size, shuffle=True)
        model.to(device, dtype=torch.float64)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        epoch_loss_sum = 0
        epoch_counter = 0
        checkpoint = {'e': 0, 'best_val': np.inf, 'best_val_epoch': np.nan}
        model.train()
        while checkpoint['e'] < args.epoch:
            # train
            for bid, x in enumerate(train_loader):
                optimizer.zero_grad()
                # masking
                obs_mask = create_mask((checkpoint['e']+1)*(args.seed+bid), (x.size(0), x.size(1))).to(device, dtype=torch.float64) # (b,d)
                x = x.to(device, dtype=torch.float64) # (b,d)
                x_mask = x * obs_mask # (b,d)
                # estimate px
                px = model.log_prob((x_mask, obs_mask))
                loss = -torch.mean(px)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
                optimizer.step()
                # update epoch loss
                epoch_loss_sum += -torch.sum(px).data.cpu().numpy()
                epoch_counter += px.size(0)
            eposs_loss = epoch_loss_sum/epoch_counter
            print(f"Epoch: [{checkpoint['e']+1}/{args.epoch}] loss: {eposs_loss}")
            if eposs_loss < checkpoint['best_val']:
                print(" [^] Best validation loss [^] ... [saving]")
                checkpoint['best_val'] = eposs_loss
                checkpoint['best_val_epoch'] = checkpoint['e'] + 1
            epoch_loss_sum = 0
            epoch_counter = 0
            checkpoint['e'] += 1
            # save checkpoints
            save_path = Path("model_save") / data_gg['data_ver'] / data_gg['data_fn']
            save_path.mkdir(parents=True, exist_ok=True)
            if checkpoint['e'] >= 4000 and checkpoint['e'] % 500 == 0:
                torch.save(model.state_dict(), (
                                                    f"model_save/"
                                                    f"{data_gg['data_ver']}/"
                                                    f"{data_gg['data_fn']}/"
                                                    f"{args.mode}_seed={args.data_seed}_cpt={checkpoint['e']}.pt"
                                                ))
    else:     
        n_inference_samples = 1000
        inference_batch_size = 6000 # change to fit memory
        masks_per_batch = inference_batch_size//n_inference_samples
        inference_samples = data["data"]
        d = args.d
        Mbs = identify_mb(data_gg['data']["dag"])
        model.load_state_dict(torch.load(
                                    (
                                        f"model_save/"
                                        f"{data_gg['data_ver']}/"
                                        f"{data_gg['data_fn']}/"
                                        f"{args.mode}_seed={args.data_seed}_cpt={args.ckps}.pt"
                                    ), weights_only=True))
        model = model.to(device)
        inferred_mbs = {X: [] for X in range(d)}
        inferred_delta = {X: [] for X in range(d)}
        with torch.no_grad():
            model.eval()
            for layer in range(1, d):
                # step 1: create combination of observed indices
                # n_combinations <= 2d^2 elements (n_combinations decreases when layer increases)
                Mb_X_Zs = set()
                X_Mb_X_Zs = set()
                for X in range(d):
                    if len(inferred_mbs[X]) == len(Mbs[X]): continue
                    for Z in range(d):
                        if Z in inferred_mbs[X] or Z == X: continue
                        Mb_X_Zs.add(tuple(inferred_mbs[X] + [Z]))
                        X_Mb_X_Zs.add(tuple([X] + inferred_mbs[X] + [Z]))

                if len(Mb_X_Zs) == 0 or len(X_Mb_X_Zs) == 0:
                    break
                # step 2: create list of masks
                Mb_X_Z_obs_indices = torch.tensor(list(map(list, Mb_X_Zs)))
                X_Mb_X_Z_obs_indices = torch.tensor(list(map(list, X_Mb_X_Zs)))
                obs_indices = [tuple(Mb_X_Z_obs_indices[i,:].tolist()) for i in range(Mb_X_Z_obs_indices.size(0))] + \
                                [tuple(X_Mb_X_Z_obs_indices[i,:].tolist()) for i in range(X_Mb_X_Z_obs_indices.size(0))]
                Mb_X_Z_masks = torch.zeros((Mb_X_Z_obs_indices.size(0), d), dtype=Mb_X_Z_obs_indices.dtype) # ((Mb_X_Z_n_combinations,d)
                Mb_X_Z_masks.scatter_(dim=1,
                                        index=Mb_X_Z_obs_indices,
                                        src=torch.ones_like(Mb_X_Z_obs_indices)) # (Mb_X_Z_n_combinations,d)
                X_Mb_X_Z_masks = torch.zeros((X_Mb_X_Z_obs_indices.size(0), d), dtype=X_Mb_X_Z_obs_indices.dtype) # ((Mb_X_Z_n_combinations,d)
                X_Mb_X_Z_masks.scatter_(dim=1,
                                        index=X_Mb_X_Z_obs_indices,
                                        src=torch.ones_like(X_Mb_X_Z_obs_indices)) # (X_Mb_X_Z_n_combinations,d)
                masks = torch.cat((Mb_X_Z_masks, X_Mb_X_Z_masks), dim=0) # (n_combinations,d)
                n_combinations = masks.size(0)
                # step 4: setup dataloader -> each loading step should parallelly process more than one mask & step 5: inference
                layer_px = torch.empty(0).to(device=device)
                # Eg., n_combinations = 6, masks_per_batch=2 -> 3 -> bidx = 0,1,2 -> 01, 23
                left_over = n_combinations - (n_combinations//masks_per_batch)*masks_per_batch
                for bidx in tqdm(range(n_combinations//masks_per_batch)):
                    x = torch.tensor(np.tile(inference_samples, (masks_per_batch, 1))).to(device,
                                                                            dtype=torch.float64)
                    # (masks_per_batch*n_inference_samples,d)
                    obs_mask = masks[bidx*masks_per_batch:(bidx+1)*masks_per_batch,:]
                    obs_mask = obs_mask.repeat_interleave(n_inference_samples, dim=0).to(device,
                                                                                        dtype=torch.float64)
                    # (masks_per_batch*n_inference_samples,d)
                    x_mask = x * obs_mask
                    bpx = model.log_prob((x_mask,
                                        obs_mask))
                    layer_px = torch.cat((layer_px,
                                        bpx)) # final shape (n_combinations*masks_per_batch,)
                if left_over > 0:
                    x = torch.tensor(np.tile(inference_samples, (left_over, 1))).to(device,dtype=torch.float64)
                    # (masks_per_batch*n_inference_samples,d)
                    obs_mask = masks[(n_combinations//masks_per_batch)*masks_per_batch:,:]
                    obs_mask = obs_mask.repeat_interleave(n_inference_samples, dim=0).to(device,
                                                                                        dtype=torch.float64)
                    # (masks_per_batch*n_inference_samples,d)
                    x_mask = x * obs_mask
                    bpx = model.log_prob((x_mask,
                                        obs_mask))
                    layer_px = torch.cat((layer_px,
                                        bpx)) # final shape (n_combinations*masks_per_batch,)
                # step 6: postprocess to return marginal entropy of each mask and save value into a dict
                # property of this dict: <key>-<value> <=> <mask>-<marginal entropy>
                layer_px_sum = layer_px.view(n_combinations,
                                            n_inference_samples).sum(dim=1) # (n_combinations,)
                layer_entropies = -layer_px_sum/n_inference_samples # (n_combinations,)
                layer_mask_entropy = {obs_indices[i]:layer_entropies[i].item() for i in range(n_combinations)}
                # step 7: run greedy step to add new variable into each markov blanket
                # if a markov blanket set of a varible reaches cardinality, stop add varible into this set
                if layer == 1: # init h|Mb as marginal entropy of every single variable
                    h_cond_Mbs = list([layer_entropies[i].item() for i in range(n_combinations) if len(obs_indices[i]) == 1])
                for X in range(d):
                    if len(inferred_mbs[X]) == len(Mbs[X]): continue
                    max_delta_Z = -math.inf
                    max_Z = -1
                    for Z in range(d):
                        if Z in inferred_mbs[X] or Z == X: continue
                        Mb_X_Z = inferred_mbs[X] + [Z] # Mb_X U {Z}
                        X_Mb_X_Z = [X] + inferred_mbs[X] + [Z] # {X} U Mb_X U {Z}
                        h_Mb_X_Z = layer_mask_entropy[tuple(Mb_X_Z)]
                        h_X_Mb_X_Z = layer_mask_entropy[tuple(X_Mb_X_Z)]
                        h_X_cond_Mb_X__Z = h_X_Mb_X_Z - h_Mb_X_Z  # h(X|Mb_X U {Z}) = h({X} U Mb_X U {Z}) - h(Mb_X U {Z})
                        delta_Z = h_cond_Mbs[X] - h_X_cond_Mb_X__Z
                        if delta_Z > max_delta_Z:
                            max_Z = Z
                            max_delta_Z = delta_Z
                    h_cond_Mbs[X] = h_cond_Mbs[X] - max_delta_Z # est h|Mb when layer > 1
                    inferred_mbs[X].append(max_Z)
                    inferred_delta[X].append(max_delta_Z)
        # evaluate
        ndcgs = {}
        for X in range(d):
            if len(Mbs[X]) > 0:
                ndcg = ndcg_at_k(list(Mbs[X]), list(inferred_mbs[X]))
                ndcgs[X] = ndcg
        print("ndcg", sum(ndcgs.values()) / len(ndcgs))
        # save result
        save_path = Path("result_save") / data_gg['data_ver'] / data_gg['data_fn']
        save_path.mkdir(parents=True, exist_ok=True)
        result = {'mb': inferred_mbs}
        with open(
            (
                f"result_save/"
                f"{data_gg['data_ver']}/"
                f"{data_gg['data_fn']}/"
                f"{args.mode}_seed={args.data_seed}_cpt={args.ckps}.pkl"
            ), 'wb') as pickle_file:
            pickle.dump(result, pickle_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=30)
    parser.add_argument('--data_seed', type=int, default=42)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epoch', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--clip', type=float, default=5.0)
    parser.add_argument('--mode', type=str, choices=['flow', 'energy'], default='flow')
    parser.add_argument('--conditioner', type=str, default='maam')
    parser.add_argument('--transformer', type=str, default='sigmoidal')
    parser.add_argument('--base', type=str, default='plain')
    parser.add_argument('--n_flows', type=int, default=1)
    parser.add_argument('--units', type=int, default=6)
    parser.add_argument('--n_outlayers', type=int, default=20)
    parser.add_argument('--ds_dim', type=int, default=4)
    parser.add_argument('--n_ds_layers', type=int, default=1)
    parser.add_argument('--train', action='store_true') # train/infer
    parser.add_argument('--ckps', type=int, default=5000)

    args = parser.parse_args()
    
    # make model configuration (flow configs by default, 
    #                           CHANGE MODEL CONFIGS HERE)
    model_config = {
        'conditioner_name': args.conditioner,
        'transformer_name': args.transformer,
        'base_dist_name': args.base, 
        'normflow': {
                        'n_flows': args.n_flows, 
                        'flip': False
        },
        'conditioner': {
                            'dim': args.d, 
                            'hid_dim': (args.d-1)*args.units, 
                            'dh_per_unit': args.d-1, 
                            'units': args.units, 
                            'num_hid_layers': 1, 
                            'num_outlayers': args.n_outlayers
        },
        # to enable compact version, assign dh_per_unit < d-1
        'transformer': {
                            'mollify': 0.0, 
                            'num_ds_dim': args.ds_dim, 
                            'num_ds_layers': args.n_ds_layers
        },
        'base_dist': {
                        'name': 'gaussian'
        }
    }

    # make train configuration
    train_config = {
                        'seed': args.seed,
                        'epoch': args.epoch,
                        'batch_size': args.batch_size,
                        'lr': args.lr,
                        'clip': args.clip
    }

    # make data configuration (CHANGE DATA CONFIGS HERE)
    data_config = {
                        'seed': args.data_seed,
                        'd': args.d,
                        'e': args.d,
                        'dag_type': 'ER',
                        'method': 'nonlinear',
                        'sem_type': 'gp',
                        'n_samples': 1000
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    # set seed 
    np.random.seed(args.seed)
    torch.manual_seed(args.seed+10000)
    random.seed(args.seed)

    # load data 
    data_fn = (
            f"data_d={data_config['d']}_"
            f"e={data_config['e']}_"
            f"N={data_config['n_samples']}_"
            f"dag={data_config['dag_type']}_"
            f"sem={data_config['sem_type']}_"
            f"seed={data_config['seed']}"
    )
    data_ver = data_config['sem_type'] + "_d" + str(args.d)
    data = np.load(f'data_gen/{data_ver}/{data_fn}.npz')
    data_gg = {
        'data': data,
        'data_fn': data_fn,
        'data_ver': data_ver,
        'data_config': data_config
    }

    # create model
    if args.mode == 'flow':
        model = AMFlow(model_config) # by default
    elif args.mode == 'energy':
        model = ACEModel(model_config)

    # train/infer
    run(args, 
        data_gg, 
        model, 
        device=device)





