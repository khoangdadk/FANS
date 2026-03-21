import numpy as np
import torch
 

 
def create_mask(seed, shape, strategy="partial", leaf=True, k=20):
    b, d = shape
    if strategy == "uniform":
        if leaf:
            sigma = torch.rand(size=(b,d-1)).argsort(dim=-1)
            t = torch.randint(low=0, high=d, size=(b,)).reshape(b, 1)
            mask = (sigma < t)
            mask = torch.cat((mask, torch.ones(b, 1, dtype=torch.float)), dim=1)
        else:
            sigma = torch.rand(size=(b,d)).argsort(dim=-1)
            t = torch.randint(low=1, high=d+1, size=(b,)).reshape(b, 1)
            mask = (sigma < t)
    elif strategy == "bernoulli":
        if leaf:
            mask = torch.from_numpy(np.random.RandomState(seed=seed).binomial(1, p=0.5, size=(b,d-1)))
            mask = torch.cat((mask, torch.ones(b, 1, dtype=torch.float)), dim=1)
        else:
            mask = torch.from_numpy(np.random.RandomState(seed=seed).binomial(1, p=0.5, size=shape))
    elif strategy == "partial":
        if leaf:
            mask = torch.zeros(b, d-1, dtype=torch.float)
            for i in range(b):
                num_ones = torch.randint(low=0, high=k+1, size=(1,)).item()
                indices = torch.randperm(d-1)[:num_ones]
                mask[i, indices] = 1.
            mask = torch.cat((mask, torch.ones(b, 1, dtype=torch.float)), dim=1)
        else:
            mask = torch.zeros(b, d, dtype=torch.float)
            for i in range(b):
                num_ones = torch.randint(low=1, high=k+1, size=(1,)).item()
                indices = torch.randperm(d)[:num_ones]
                mask[i, indices] = 1.
    return mask