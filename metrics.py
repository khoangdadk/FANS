import numpy as np
from sklearn import metrics 



def dcg_at_k(relevance_scores):
    """
    Calculate Discounted Cumulative Gain (DCG) at rank k.

    Parameters:
        relevance_scores (list or array): Relevance scores in the order of the predicted ranking.
        k (int): Rank position to calculate DCG up to.

    Returns:
        float: DCG score at k.
    """
    relevance_scores = np.asarray(relevance_scores)
    if relevance_scores.size == 0:
        return 0.0
    # Compute DCG
    return np.sum(relevance_scores / np.log2(np.arange(2, relevance_scores.size + 2)))



def ndcg_at_k(true_labels, pred_labels):
    """
    Calculate Normalized Discounted Cumulative Gain (nDCG) at rank k.

    Parameters:
        true_labels (list): List of relevant items (ground truth).
        predicted_ranking (list): List of predicted items (ranked).
        k (int): Rank position to calculate nDCG up to.

    Returns:
        float: nDCG score at k.
    """
    k = min(len(true_labels), len(pred_labels))
    # Binary relevance: 1 if an item in predicted ranking is in true labels, 0 otherwise
    relevance_scores = [1 if item in true_labels else 0 for item in pred_labels[:k]]

    # Compute DCG for the predicted ranking
    dcg = dcg_at_k(relevance_scores)

    # Compute the ideal DCG (IDCG) for the ground truth
    ideal_relevance_scores = [1] * len(true_labels)

    idcg = dcg_at_k(ideal_relevance_scores)

    # Avoid division by zero
    if idcg == 0:
        return 0.0

    # Compute nDCG, recall
    return dcg / idcg



def AveP(true_list, pred_list):
    n = min(len(pred_list), len(true_list))
    try:
        precision_at_i = [sum(1 for label in pred_list[:k] if label in true_list) / k for k in range(1,n+1)]
        average_precision = sum(precision_at_i[j] for j in range(n) if pred_list[j] in true_list) / len(true_list)
        return average_precision
    except:
        return 0.0



def f1(true_blanket, predicted_blanket):
    true_positives = len(predicted_blanket & true_blanket)
    precision = true_positives / len(predicted_blanket) if len(predicted_blanket) > 0 else 0.0
    recall = true_positives / len(true_blanket)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1



def identify_mb_each_node(adj_mat: np.ndarray, node: int) -> set:
    if adj_mat.shape[0] != adj_mat.shape[1]:
        raise ValueError("Adjacency matrix must be square")
    if node < 0 or node >= adj_mat.shape[0]:
        raise ValueError("Node index out of bounds")

    n = adj_mat.shape[0]
    markov_blanket = set()

    # Find parents (nodes that have edges pointing to our target node)
    parents = {i for i in range(n) if int(adj_mat[i, node]) == 1}

    # Find children (nodes that our target node points to)
    children = {i for i in range(n) if int(adj_mat[node, i]) == 1}

    # Find spouses (parents of children)
    spouses = set()
    for child in children:
        child_parents = {i for i in range(n) if int(adj_mat[i, child]) == 1}
        spouses.update(child_parents)

    # Combine all nodes (excluding the target node itself)
    markov_blanket = parents | children | spouses - {node}

    return markov_blanket



def edge_auroc(pred_edges: np.ndarray, true_edges: np.ndarray):
    if true_edges.min() < 0 or true_edges.max() > 1:
        print("Groundtruth is CPDAG")
        true_edges = np.clip(true_edges, 0, 1)
    if pred_edges.min() < 0 or pred_edges.max() > 1:
        pred_edges = np.clip(pred_edges, 0, 1)
    fpr, tpr, _ = metrics.roc_curve(true_edges.reshape(-1), pred_edges.reshape(-1))
    auc = metrics.auc(fpr, tpr)
    return auc



def edge_apr(pred_edges: np.ndarray, true_edges: np.ndarray):
    if true_edges.min() < 0 or true_edges.max() > 1:
        print("Groundtruth is CPDAG")
        true_edges = np.clip(true_edges, 0, 1)
    if pred_edges.min() < 0 or pred_edges.max() > 1:
        pred_edges = np.clip(pred_edges, 0, 1)
    return metrics.average_precision_score(true_edges.reshape(-1), pred_edges.reshape(-1))



# https://github.com/xunzheng/notears/blob/master/notears/utils.py
def SHD(A_true: np.ndarray, A_pred: np.ndarray, **kwargs):
    # linear index of nonzeros
    pred = np.flatnonzero(A_pred == 1)
    cond = np.flatnonzero(A_true)
    cond_reversed = np.flatnonzero(A_true.T)
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(A_pred + A_pred.T))
    cond_lower = np.flatnonzero(np.tril(A_true + A_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return shd, len(extra_lower), len(missing_lower), len(reverse)



def SHDs(A_true, A_preds):
    shds = []
    for A_pred in A_preds:
        A_pred = A_pred.astype(int)
        shd = SHD(A_true, A_pred)[0]
        shds.append(shd)
    return shds


