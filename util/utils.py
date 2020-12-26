'''Utils function'''
import torch
import torch.nn as nn


def l1norm(X, dim=1, eps=1e-12):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X

def l2norm(X, dim=-1, eps=1e-12):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def lambda_sigmoid(x, lamda=5):
    return 1 / ( 1 + torch.exp(-lamda * x) )

def min_max_resacle(x, eps=1e-12):
    return torch.div(x - x.min(), x.max() - x.min() + eps)

def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())

def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score

def Cos_dis(x):
    """
    Calculate the Cosine distance among each raw of x
    :param x: N X D
                N: the object number
                D: Dimension of the feature
    :return: N X N distance matrix
    """
    dist_mat = cosine_sim(x, x)
    dist_mat = torch.max(dist_mat, dist_mat.t())
    return dist_mat


def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=False, m_prob=1, return_index=False):
    """
    :param dis_mat: node distance matrix
    :param k_neig: K nearest neighbor
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :param return_index: index of K-nearest neigbour nodes
    :return: N_object X N_hyperedge
    """
    if return_index == False:
        n_obj = dis_mat.shape[0]
        n_edge = n_obj
        H = torch.zeros((n_obj, n_edge)).cuda()
        for center_idx in range(n_obj):
            dis_mat[center_idx, center_idx] = 0
            dis_vec = dis_mat[center_idx]
            _, nearest_idx = torch.sort(dis_vec, -1, descending=True)   # from large to small (2. for cosine dist)
            avg_dis = torch.mean(dis_vec)

            if not (nearest_idx[:k_neig] == center_idx).any():
                nearest_idx[k_neig - 1] = center_idx
            for node_idx in nearest_idx[:k_neig]:
                if is_probH:
                    H[node_idx, center_idx] = torch.exp(-1 * ((m_prob * avg_dis) ** 2) / (dis_vec[node_idx] ** 2 )  )  # for cosine distance (^2)
                else:
                    H[node_idx, center_idx] = 1.0
        return H
    ## return index of K-nearest neigbour nodes
    else:
        n_obj = dis_mat.shape[0]
        index_K_neighbour = torch.zeros((n_obj, k_neig))

        for center_idx in range(n_obj):
            dis_mat[center_idx, center_idx] = 0
            dis_vec = dis_mat[center_idx]
            _, nearest_idx = torch.sort(dis_vec, -1, descending=True)   # from large to small (2. for cosine dist)
            index_K_neighbour[center_idx] = nearest_idx[:k_neig]

        return index_K_neighbour


def construct_H_with_KNN(X, K_neigs=[10], split_diff_scale=False, is_probH=True, m_prob=1, return_index=False):
    """
    :param X: N_object x feature_number (N_nodes, emb_dim)
    :param K_neigs: the number of neighbor expansion
    :param split_diff_scale: whether split hyperedge group at different neighbor scale
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object x N_hyperedge
    """
    if len(X.shape) != 2:
        X = X.reshape(-1, X.shape[-1])
    if type(K_neigs) == int:
        K_neigs = [K_neigs]
    dis_mat = Cos_dis(X)    # 2) Cosine distance
    H = []
    for k_neig in K_neigs:

        if return_index == False:
            H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
            if split_diff_scale:
                H.append(H_tmp)
            else:
                H = H_tmp
        else:
            index_KNN = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob, return_index)
            H = index_KNN
    return H







