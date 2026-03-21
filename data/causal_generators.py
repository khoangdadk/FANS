# https://github.com/huawei-noah/trustworthyAI/tree/master/gcastle/castle/datasets

import logging
import random
from random import sample
from copy import deepcopy
import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
from scipy.special import expit as sigmoid



def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)



def generate_dag(d, e, dag_type, seed=None, weight_range=(0.5, 2), **kwargs):
    dag_generators = {
        'ER': DAG.erdos_renyi,
        'SF': DAG.scale_free
    }
    dag_gen = dag_generators[dag_type]
    W = dag_gen(n_nodes=d, 
                n_edges=e, 
                seed=seed, 
                weight_range=weight_range)
    Adj = (W != 0).astype(int)
    return Adj



def create_causal_data(N, d, e, dag_type, method, sem_type, noise_scale, weight_range=(0.5, 2), create_test=False, seed=None, mix_noise=False):
    """
    N: number of samples
    d: number of nodes
    e: number of expected edges
    dag_type: 'ER' or 'SF'
    method: 'linear' or 'nonlinear'
    sem_type: for linear, this is the noise type (gauss, exp, gumbel, uniform, laplace, mix)
            for nonlinear, this is the form of function (m)
    noise_scale: scale of noise
    seed: for randomness
    ----
    return the adjacency matrix and the generated data
    """
    dag_generators = {
        'ER': DAG.erdos_renyi,
        'SF': DAG.scale_free
    }
    dag_gen = dag_generators[dag_type]
    Adj = dag_gen(n_nodes=d, 
                n_edges=e, 
                seed=seed, 
                weight_range=weight_range)
    train = CausalDataset(Adj, n=N, 
                      method=method,
                      sem_type=sem_type,
                      noise_scale=noise_scale,
                      seed=seed,
                      mix_noise=mix_noise)
    if create_test:
        test = CausalDataset(Adj, n=1000, method=method, sem_type=sem_type, noise_scale=noise_scale)
        return train.B, train.X.astype(np.float32), test.X.astype(np.float32)
    else:
        return train.B, train.X.astype(np.float32)



class DAG(object):
    '''
    A class for simulating random (causal) DAG, where any DAG generator
    method would return the weighed/binary adjacency matrix of a DAG.
    Besides, we recommend using the python package "NetworkX"
    to create more structures types.
    '''

    @staticmethod
    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    @staticmethod
    def _random_acyclic_orientation(B_und):
        B = np.tril(DAG._random_permutation(B_und), k=-1)
        B_perm = DAG._random_permutation(B)
        return B_perm

    @staticmethod
    def _graph_to_adjmat(G):
        return nx.to_numpy_array(G)

    @staticmethod
    def _BtoW(B, d, w_range):
        U = np.random.uniform(low=w_range[0], high=w_range[1], size=[d, d])
        U[np.random.rand(d, d) < 0.5] *= -1
        W = (B != 0).astype(float) * U
        return W

    @staticmethod
    def _low_rank_dag(d, degree, rank):
        """
        Simulate random low rank DAG with some expected degree.

        Parameters
        ----------
        d: int
            Number of nodes.
        degree: int
            Expected node degree, in + out.
        rank: int
            Maximum rank (rank < d-1).

        Return
        ------
        B: np.nparray
            Initialize DAG.
        """
        prob = float(degree) / (d - 1)
        B = np.triu((np.random.rand(d, d) < prob).astype(float), k=1)
        total_edge_num = np.sum(B == 1)
        sampled_pa = sample(range(d - 1), rank)
        sampled_pa.sort(reverse=True)
        sampled_ch = []
        for i in sampled_pa:
            candidate = set(range(i + 1, d))
            candidate = candidate - set(sampled_ch)
            sampled_ch.append(sample(candidate, 1)[0])
            B[i, sampled_ch[-1]] = 1
        remaining_pa = list(set(range(d)) - set(sampled_pa))
        remaining_ch = list(set(range(d)) - set(sampled_ch))
        B[np.ix_(remaining_pa, remaining_ch)] = 0

        # mask B
        maskedB = B + np.tril(np.ones((d, d)))
        maskedB[np.ix_(remaining_pa, remaining_ch)] = 1
        B[maskedB == 0] = 1

        remaining_ch_set = set([i + d for i in remaining_ch])
        remaining_pa_set = set(remaining_pa)

        edges = np.transpose(np.nonzero(B))
        edges[:, 1] += d
        bigraph = nx.Graph()
        bigraph.add_nodes_from(range(2 * d))
        bigraph.add_edges_from(edges)
        M = nx.bipartite.maximum_matching(bigraph, top_nodes=range(d))
        while len(M) > 2 * rank:
            keys = set(M.keys())
            rmv_cand = keys & (remaining_pa_set | remaining_ch_set)
            p = sample(rmv_cand, 1)[0]
            c = M[p]
            # destroy p-c
            bigraph.remove_edge(p, c)
            M = nx.bipartite.maximum_matching(bigraph, top_nodes=range(d))

        new_edges = np.array(bigraph.edges)
        for i in range(len(new_edges)):
            new_edges[i,].sort()
        new_edges[:, 1] -= d

        BB = np.zeros((d, d))
        B = np.zeros((d, d))
        BB[new_edges[:, 0], new_edges[:, 1]] = 1

        if np.sum(BB == 1) > total_edge_num:
            delta = total_edge_num - rank
            BB[sampled_pa, sampled_ch] = 0
            rmv_cand_edges = np.transpose(np.nonzero(BB))
            if delta <= 0:
                raise RuntimeError(r'Number of edges is below the rank, please \
                                   set a larger edge or degree \
                                   (you can change seed or increase degree).')
            selected = np.array(sample(rmv_cand_edges.tolist(), delta))
            B[selected[:, 0], selected[:, 1]] = 1
            B[sampled_pa, sampled_ch] = 1
        else:
            B = deepcopy(BB)

        B = B.transpose()
        return B

    @staticmethod
    def erdos_renyi(n_nodes, n_edges, weight_range=None, seed=None):

        assert n_nodes > 0
        set_random_seed(seed)
        # Erdos-Renyi
        creation_prob = (2 * n_edges) / ((n_nodes - 1) * n_nodes)
        G_und = nx.erdos_renyi_graph(n=n_nodes, p=creation_prob, seed=seed)
        B_und = DAG._graph_to_adjmat(G_und)
        B = DAG._random_acyclic_orientation(B_und)
        if weight_range is None:
            return B
        else:
            W = DAG._BtoW(B, n_nodes, weight_range)
        return W

    @staticmethod
    def scale_free(n_nodes, n_edges, weight_range=None, seed=None):

        assert (n_nodes > 0 and n_edges >= n_nodes and n_edges < n_nodes * n_nodes)
        set_random_seed(seed)
        # Scale-free, Barabasi-Albert
        m = int(round(n_edges / n_nodes))
        G_und = nx.barabasi_albert_graph(n=n_nodes, m=m)
        B_und = DAG._graph_to_adjmat(G_und)
        B = DAG._random_acyclic_orientation(B_und)
        if weight_range is None:
            return B
        else:
            W = DAG._BtoW(B, n_nodes, weight_range)
        return W

    @staticmethod
    def bipartite(n_nodes, n_edges, split_ratio = 0.2, weight_range=None, seed=None):

        assert n_nodes > 0
        set_random_seed(seed)
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        n_top = int(split_ratio * n_nodes)
        n_bottom = n_nodes -  n_top
        creation_prob = n_edges/(n_top*n_bottom)
        G_und = bipartite.random_graph(n_top, n_bottom, p=creation_prob, directed=True)
        B_und = DAG._graph_to_adjmat(G_und)
        B = DAG._random_acyclic_orientation(B_und)
        if weight_range is None:
            return B
        else:
            W = DAG._BtoW(B, n_nodes, weight_range)
        return W

    @staticmethod
    def hierarchical(n_nodes, degree=5, graph_level=5, weight_range=None, seed=None):

        assert n_nodes > 1
        set_random_seed(seed)
        prob = float(degree) / (n_nodes - 1)
        B = np.tril((np.random.rand(n_nodes, n_nodes) < prob).astype(float), k=-1)
        point = sample(range(n_nodes - 1), graph_level - 1)
        point.sort()
        point = [0] + [x + 1 for x in point] + [n_nodes]
        for i in range(graph_level):
            B[point[i]:point[i + 1], point[i]:point[i + 1]] = 0
        if weight_range is None:
            return B
        else:
            W = DAG._BtoW(B, n_nodes, weight_range)
        return W

    @staticmethod
    def low_rank(n_nodes, degree=1, rank=5, weight_range=None, seed=None):

        assert n_nodes > 0
        set_random_seed(seed)
        B = DAG._low_rank_dag(n_nodes, degree, rank)
        if weight_range is None:
            return B
        else:
            W = DAG._BtoW(B, n_nodes, weight_range)
        return W



class CausalDataset(object):
    '''
    Simulate IID datasets for causal structure learning.

    Parameters
    ----------
    W: np.ndarray
        Weighted adjacency matrix for the target causal graph.
    n: int
        Number of samples for standard trainning dataset.
    method: str, (linear or nonlinear), default='linear'
        Distribution for standard trainning dataset.
    sem_type: str
        gauss, exp, gumbel, uniform, logistic (linear); 
        mlp, mim, gp, gp-add, quadratic (nonlinear).
    noise_scale: float
        Scale parameter of noise distribution in linear SEM.
    '''

    def __init__(self, W, n=1000, method='linear', 
                 sem_type='gauss', noise_scale=1.0, seed=42, mix_noise=False):

        self.B = (W != 0).astype(int)
        if method == 'linear':
            self.X = CausalDataset._simulate_linear_sem(
                    W, n, sem_type, noise_scale, seed, mix_noise)
        elif method == 'nonlinear':
            self.X = CausalDataset._simulate_nonlinear_sem(
                    W, n, sem_type, noise_scale, seed, mix_noise)
        logging.info('Finished synthetic dataset')

    @staticmethod
    def _simulate_linear_sem(W, n, sem_type, noise_scale, seed, mix_noise):
        """
        Simulate samples from linear SEM with specified type of noise.
        For uniform, noise z ~ uniform(-a, a), where a = noise_scale.
        """
        def _simulate_single_equation(X, w, scale, j):
            """X: [n, num of parents], w: [num of parents], x: [n]"""
            if sem_type == 'gauss':
                z = np.random.normal(scale=scale, size=n)
                x = X @ w + z
            elif sem_type == 'exp':
                z = np.random.exponential(scale=scale, size=n)
                x = X @ w + z
            elif sem_type == 'gumbel':
                z = np.random.gumbel(scale=scale, size=n)
                x = X @ w + z
            elif sem_type == 'uniform':
                z = np.random.uniform(low=-scale, high=scale, size=n)
                x = X @ w + z
            elif sem_type == 'laplace':
                z = np.random.laplace(scale=scale, size=n)
                x = X @ w + z
            elif sem_type == 'mix':
                noise_types = [gumbel_noise, laplace_noise, uniform_noise, gaussian_noise, exp_noise]
                rng = np.random.RandomState(seed + j)
                noise_func = noise_types[rng.randint(len(noise_types))] 
                scale = random.uniform(0.5, 1.5)
                z = noise_func(scale, n)
                x = X @ w + z
            else: 
                raise ValueError('Unknown sem type. In a linear model, \
                                 the options are as follows: gauss, exp, \
                                 gumbel, uniform, mix.')
            return x
        d = W.shape[0]
        if noise_scale is None:
            scale_vec = np.ones(d)
        elif np.isscalar(noise_scale):
            scale_vec = noise_scale * np.ones(d)
        else:
            if len(noise_scale) != d:
                raise ValueError('noise scale must be a scalar or has length d')
            scale_vec = noise_scale
        G_nx =  nx.from_numpy_array(W, create_using=nx.DiGraph)
        if not nx.is_directed_acyclic_graph(G_nx):
            raise ValueError('W must be a DAG')
        if np.isinf(n):  # population risk for linear gauss SEM
            if sem_type == 'gauss':
                # make 1/d X'X = true cov
                X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
                return X
            else:
                raise ValueError('population risk not available')
        # empirical risk
        ordered_vertices = list(nx.topological_sort(G_nx))
        assert len(ordered_vertices) == d
        X = np.zeros([n, d])
        for j in ordered_vertices:
            parents = list(G_nx.predecessors(j))
            X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j], j)
        return X

    @staticmethod
    def _simulate_nonlinear_sem(W, n, sem_type, noise_scale, seed, mix_noise):
        """
        Simulate samples from nonlinear SEM.
        """

        rng = np.random.RandomState(seed)
        noise_types = [gumbel_noise, laplace_noise, uniform_noise, gaussian_noise, exp_noise]

        def _simulate_single_equation(X, scale, j):
            """X: [n, num of parents], x: [n]"""
            
            if mix_noise:
                noise_func = noise_types[rng.randint(len(noise_types))] 
                # scale = random.uniform(0.5, 2)
                z = noise_func(scale=scale, size=n)
            else:
                z = np.random.normal(scale=scale, size=n)

            pa_size = X.shape[1]
            if pa_size == 0:
                return z
            if sem_type == 'nonlin':
                hidden = 32
                W11 = rng.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
                W12 = rng.uniform(low=0.5, high=2.0, size=hidden)
                W11[rng.rand(*W11.shape) < 0.5] *= -1
                W12[rng.rand(hidden) < 0.5] *= -1
                W21 = rng.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
                W22 = rng.uniform(low=0.5, high=2.0, size=hidden)
                W21[rng.rand(*W21.shape) < 0.5] *= -1
                W22[rng.rand(hidden) < 0.5] *= -1
                x = np.cos(np.pi * X @ W11) @ W12 + np.sin(np.pi * X @ W21) @ W22 + z
            elif sem_type == 'mlp':
                hidden = 100
                W1 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
                W1[np.random.rand(*W1.shape) < 0.5] *= -1
                W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
                W2[np.random.rand(hidden) < 0.5] *= -1
                x = sigmoid(X @ W1) @ W2 + z
            elif sem_type == 'postnonlin':
                hidden = 32
                np.random.seed(seed + j)
                W11 = np.random.uniform(low=0.1, high=1.0, size=[pa_size, hidden])
                W12 = np.random.uniform(low=0.1, high=1.0, size=hidden)
                W11[np.random.rand(*W11.shape) < 0.5] *= -1
                W12[np.random.rand(hidden) < 0.5] *= -1
                f_types = [sin_func, cos_func, polynomial_func, arctan_func]
                f_func, g_func = rng.choice(f_types, size=2, replace=False).tolist()
                x = g_func(f_func(X @ W11) @ W12 + z)
            elif sem_type == 'gp':
                from sklearn.gaussian_process import GaussianProcessRegressor
                gp = GaussianProcessRegressor()
                x = gp.sample_y(X, random_state=None).flatten() + z
            else:
                raise ValueError('Unknown sem type. In a nonlinear model, \
                                 the options are as follows: nonlin or postnonlin.')
            return x

        B = (W != 0).astype(int)
        d = B.shape[0]
        if noise_scale is None:
            scale_vec = np.ones(d)
        elif np.isscalar(noise_scale):
            scale_vec = noise_scale * np.ones(d)
        else:
            if len(noise_scale) != d:
                raise ValueError('noise scale must be a scalar or has length d')
            scale_vec = noise_scale

        X = np.zeros([n, d])
        G_nx =  nx.from_numpy_array(B, create_using=nx.DiGraph)
        ordered_vertices = list(nx.topological_sort(G_nx))
        assert len(ordered_vertices) == d
        from tqdm import tqdm
        for j in tqdm(ordered_vertices):
            parents = list(G_nx.predecessors(j))
            X[:, j] = _simulate_single_equation(X[:, parents], scale_vec[j], j)
        return X




def polynomial_func(x):
    return x - 1/2 * x**2

def arctan_func(x):
    return 2/np.pi * np.arctan(x)

def sin_func(x):
    return np.sin(np.pi * x)

def cos_func(x):
    return np.cos(np.pi * x)

def uniform_noise(scale, size):
    return np.random.uniform(low=-scale, high=scale, size=size)

def laplace_noise(scale, size):
    return np.random.laplace(scale=scale, size=size)

def gaussian_noise(scale, size):
    return np.random.normal(scale=scale, size=size)

def gumbel_noise(scale, size):
    return np.random.gumbel(scale=scale, size=size)

def exp_noise(scale, size):
    return np.random.exponential(scale=scale, size=size)
