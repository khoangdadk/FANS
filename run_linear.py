import os
import math
import numpy as np
import statistics
from tqdm import tqdm
from joblib import Parallel, delayed
from metrics import ndcg_at_k, f1, AveP
from utils.graph import symmetrize_markov_blanket, identify_mb



def linear_gaussian_entropy(cov_mat, vars):
    if isinstance(vars, list):
        return 0.5 * (len(vars) * np.log(2 * np.pi * np.e) + np.log(np.linalg.det(cov_mat[np.ix_(vars,vars)])))
    return 0.5 * (np.log(2 * np.pi * np.e) + np.log(cov_mat[vars,vars]))



def fans_linear(X, d, p=15, gep=0.005, sep=0.002):        

    '''
    Theorem: S=Mb_x is the minimizer of h(x|S) 
    The conditional entropy is calculated based on the difference between two marginal entropies
    h(x|S) = h(x,S) - h(S)
    '''

    Mb_X = []
    invalid_nodes = []

    '''
        grow pass: greedily add variables that maximize the decrease of conditional entropy
        return a markov blanket of X with an arbitrary size 
    '''

    h_X_cond_Mb_X = linear_gaussian_entropy(cov_mat, [X])  # estimate h(X|Mb_X)
    invalid_streak = 0
    traverse = 0

    while True:
        max_delta_Z = -math.inf
        max_Z = -1
        # estimate h(X|Mb_X U {Z})
        Z_set = [e for e in range(d) if e not in Mb_X and e != X]  
        for Z in Z_set:
            # h(X|Mb_X U {Z}) = h({X} U Mb_X U {Z}) - h(Mb_X U {Z})
            Mb_X_Z = Mb_X + [Z]
            X_Mb_X_Z = [X] + Mb_X + [Z]
            h_Mb_X_Z = linear_gaussian_entropy(cov_mat, Mb_X_Z)
            h_X_Mb_X_Z = linear_gaussian_entropy(cov_mat, X_Mb_X_Z)
            h_X_cond_Mb_X__Z = h_X_Mb_X_Z - h_Mb_X_Z
            delta_Z = h_X_cond_Mb_X - h_X_cond_Mb_X__Z
            if delta_Z > max_delta_Z:
                max_Z = Z
                max_delta_Z = delta_Z
        h_X_cond_Mb_X = h_X_cond_Mb_X - max_delta_Z
        if max_delta_Z < gep:
            invalid_nodes.append(max_Z)
            invalid_streak += 1
            if invalid_streak == p: break
        else:
            invalid_streak = 0
            invalid_nodes = []
        Mb_X.append(max_Z)
        traverse += 1
        if traverse == d-1: break

    '''
        shrink pass: greedily remove variables that minimize the increase of conditional entropy
        return the markov boundary by shrinking the markov blanket from grow pass into a possibly minimal size 
    '''

    Mb_X = [e for e in Mb_X if e not in invalid_nodes]
    h_cond_Mb = linear_gaussian_entropy(cov_mat, [X] + Mb_X) - linear_gaussian_entropy(cov_mat, Mb_X) # estimate h(X|Mb_X) 

    while True:
        min_delta_Z = math.inf
        min_Z = -1
        # estimate h(X|Mb_X \ {Z})
        for Z in Mb_X:
            # h(X|(Mb_X \ {Z})) = h({X} U (Mb_X \ {Z})) - h(Mb_X \ {Z})
            Mb_X_minus_Z = [e for e in Mb_X if e != Z]
            X_Mb_X_minus_Z = [X] + Mb_X_minus_Z
            h_X_Mb_X_minus_Z = linear_gaussian_entropy(cov_mat, X_Mb_X_minus_Z)
            h_Mb_X_minus_Z = linear_gaussian_entropy(cov_mat, Mb_X_minus_Z)
            h_X_cond_Mb_X__minus_Z = h_X_Mb_X_minus_Z - h_Mb_X_minus_Z
            # delta = h(X|(Mb_X \ {Z})) - h(X|Mb_X) 
            delta_Z = h_X_cond_Mb_X__minus_Z - h_cond_Mb
            if delta_Z < min_delta_Z:
                min_Z = Z
                min_delta_Z = delta_Z
        h_cond_Mb = h_cond_Mb + min_delta_Z # est h|Mb 
        if min_delta_Z > sep or len(Mb_X) == 1: break
        Mb_X.remove(min_Z)

    return Mb_X

if __name__ == "__main__":
    
    ndcg_lst = []
    avep_lst = []
    f1_lst = []

    # RUN LINEAR GAUSSIAN DATA, D=100, E=100 (A DAG WITH 100 NODES AND 100 EDGES)
    for data_seed in [42, 123, 1234, 12345, 123456]: 
        # load test data
        data_dict = np.load(os.path.join("data_gen", "linear_d100", f'data_d={100}_e={100}_N={5000}_dag={"ER"}_sem={"gauss"}_seed={data_seed}.npz'), allow_pickle=True)
        data = data_dict["data"]
        dag = data_dict["dag"]

        cov_mat = np.cov(data, rowvar=False) # estimate the covariance matrix
        _, d = data.shape

        true_Mbs = identify_mb(dag) # find true Markov boundary from DAG

        pred_mbs = {}
        mb_infer_results = Parallel(n_jobs=-1)(delayed(fans_linear)(X, d) for X in tqdm(range(d))) # parallelly infer Markov Boundary from data 
        for X, mb in zip(range(d), mb_infer_results):
            pred_mbs[X] = mb
    
        pred_mbs = symmetrize_markov_blanket(pred_mbs, redundant=True) # ensure that X in MB of Y <=> Y in MB of X

        ndcgs = {}
        f1s = {}
        avep = {}
        for X in range(d):
            if len(true_Mbs[X]) > 0:
                f1_score = f1(true_blanket=true_Mbs[X], predicted_blanket=set(pred_mbs[X]))
                f1s[X] = f1_score
                ndcgs[X] = ndcg_at_k(list(true_Mbs[X]), list(pred_mbs[X]))
                avep[X] = AveP(list(true_Mbs[X]), list(pred_mbs[X]))
   
        f1_lst.append(sum(f1s.values()) * 100. / len(f1s))
        ndcg_lst.append(sum(ndcgs.values()) * 100. / len(ndcgs))
        avep_lst.append(sum(avep.values()) * 100. / len(avep))
    
    # evaluate
    print("ndcg", statistics.mean(ndcg_lst), statistics.stdev(ndcg_lst))
    print("avep", statistics.mean(avep_lst), statistics.stdev(avep_lst))
    print("f1", statistics.mean(f1_lst), statistics.stdev(f1_lst))
        










