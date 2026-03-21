from .utils import Argument
from .causal_generators import create_causal_data
from .masking import create_mask

def generate_data(data, args_dict):
    args = Argument(**args_dict)
    data = {}
    for seed in args.seed:
        Adj, samples = create_causal_data(N=args.n_samples, d=args.d, e=args.e, 
                dag_type=args.dag_type, method=args.method, sem_type=args.sem_type, 
                noise_scale=1.0, weight_range=(0.5, 2), seed=seed, mix_noise=args.mix_noise)
        data[seed] = (Adj, samples)
    return data