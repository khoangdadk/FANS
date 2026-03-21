import torch

 

def marginal_entropy(d, model, dataloader, device):
    def _marginal_entropy(indices):
        mask = torch.zeros(d, dtype=torch.float64) 
        mask[indices] = 1.0
        return marginal_entropy_normflow(model, dataloader, device, mask)
    return _marginal_entropy



def marginal_entropy_energy(model, dataloader, device, mask):
    # -logp(x2,x3) = -logp(x2) - logp(x3|x2) [0 1 1 1]
    # x2, x3, x4 = x2 x3|x2 x4|x2,x3
    # log(x2) -> obs_mask: [0 0 0 0]
    # log(x3|x2) -> obs_mask: [0 1 0 0]
    # log(x4|x2,x3) -> obs_mask: [0 1 1 0]
    # x1, x3, x4 = x1 x3|x1 x4|x1,x3 [1 0 1 1]
    # log(x1) -> obs_mask: [0 0 0 0]
    # log(x3|x1) -> obs_mask: [1 0 0 0]
    # log(x4|x1,x3) -> obs_mask: [1 0 1 0]
    marginal_indices = (mask != 0).nonzero().squeeze(1).tolist()[::-1]
    marginal_entropy = 0.0
    for idx in marginal_indices:
        mask[idx] = 0.0
        energy_hsum = 0
        counter = 0
        for x in dataloader:            
            energy_ll, _ = model(x.to(device, dtype=torch.float64), 
                                 mask.repeat(x.size(0), 1).to(device, dtype=torch.float64))
            energy_hsum += -torch.sum(energy_ll, dim=0)
            counter += x.size(0)
        marginal_entropy += (energy_hsum / float(counter))[idx].item()
    
    return marginal_entropy

 
 
def marginal_entropy_normflow(model, dataloader, device, mask):
    hsum = 0 
    counter = 0
    for x in dataloader:
        x = x.to(device, dtype=torch.float64)
        obs_mask = mask.repeat(x.size(0), 1).to(device, dtype=torch.float64)
        x_mask = x * obs_mask # (b,d)
        if model.get_config()["conditioner_name"] == "gru":
            x_mask = x_mask.gather(1, torch.sort((x_mask != 0) * 1, dim=1, descending=True)[1])
            obs_mask = obs_mask.gather(1, torch.sort((obs_mask != 0) * 1, dim=1, descending=True)[1])
        px = model.log_prob((x_mask, obs_mask))  
        hsum += -torch.sum(px).data.cpu().numpy()
        counter += px.size(0)
    return hsum / float(counter)
