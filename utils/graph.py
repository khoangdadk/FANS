import numpy as np



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



def identify_mb(adj_mat: np.ndarray):
    Mbs = []
    n = adj_mat.shape[0]
    for i in range(n):
        Mbs.append(identify_mb_each_node(adj_mat, i))
    return Mbs



def symmetrize_markov_blanket(mb, redundant=False):
    # Make a copy to avoid modifying during iteration
    symmetrized_mb_dict = {k: v for k, v in mb.items()}
    for node, blanket in mb.items():
        for neighbor in blanket:
            # Check if neighbor has node in its MB
            if node not in mb.get(neighbor, []):
                if not redundant:
                    symmetrized_mb_dict[node] = [e for e in symmetrized_mb_dict[node] if e != neighbor]
                else:
                    if neighbor not in symmetrized_mb_dict: symmetrized_mb_dict[neighbor] = []
                    symmetrized_mb_dict[neighbor].append(node)
    return symmetrized_mb_dict
 