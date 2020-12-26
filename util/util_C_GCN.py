import math
from urllib.request import urlretrieve
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import random
import torch.nn.functional as F


'''gen_A: co-occur matrix generation'''
def gen_A(num_classes, t, adj_file):
    import pickle
    result = pickle.load(open(adj_file, 'rb'))

    _adj = result['adj']    # (ndarray) (300, 300), count the co-accur numbers for each word in vocab
    _nums = result['nums']   # (ndarray) (300), count the total emerging numbers for each word in vocab

    # turn mat to binary according to threshold t (default t=0.4)
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1

    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj + np.identity(num_classes, np.int)   # identity square matrix
    return _adj


''' define concept adj_matrix'''
def gen_A_concept(num_classes, t, adj_file, gen_mode='ML_GCN'):
    import pickle
    result = pickle.load(open(adj_file, 'rb'))

    _nums = result['nums']
    _nums = _nums[:, np.newaxis]

    # smooth normalized adj matrix: _A_adj
    _A_adj = {}

    for key, value in result.items():
        if key == 'adj_O_P':
            _adj_OPC = result['adj_O_P']
            _adj_OPC = _adj_OPC / _nums
            # only eq.(3) in paper
            if gen_mode == 'ML_GCN':
                '''ML_GCN method'''
                _adj_OPC[_adj_OPC < t] = 0
                _adj_OPC[_adj_OPC >= t] = 1
                _adj_OPC = _adj_OPC * 0.25 / (_adj_OPC.sum(0, keepdims=True) + 1e-6)
            # only eq.(2) in paper
            elif gen_mode == 'My_rescale':
                '''Use My rescale function'''
                _adj_OPC = rescale_adj_matrix(_adj_OPC)   # rescale function eq.(2)
            # combine eq.(2) and (3) in paper
            elif gen_mode == 'Complex':
                _adj_OPC = rescale_adj_matrix(_adj_OPC)  # rescale function eq.(2)
                _adj_OPC[_adj_OPC < t] = 0
                _adj_OPC[_adj_OPC >= t] = 1
                _adj_OPC = _adj_OPC * 0.25 / (_adj_OPC.sum(0, keepdims=True) + 1e-6)  #

            _adj_OPC = _adj_OPC / (_adj_OPC.sum(0, keepdims=True) + 1e-8)
            _adj_OPC = _adj_OPC + np.identity(num_classes, np.int)  # identity square matrix
            _A_adj['adj_O_P'] = _adj_OPC

        elif key == 'adj_O_M':
            _adj_OMC = result['adj_O_M']
            _adj_OMC = _adj_OMC / _nums
            # only eq.(3) in paper
            if gen_mode == 'ML_GCN':
                '''ML_GCN method'''
                _adj_OMC[_adj_OMC < t] = 0
                _adj_OMC[_adj_OMC >= t] = 1
                _adj_OPC = _adj_OMC * 0.25 / (_adj_OMC.sum(0, keepdims=True) + 1e-6)  #
            # only eq.(2) in paper
            elif gen_mode == 'My_rescale':
                '''Use My rescale function'''
                _adj_OMC = rescale_adj_matrix(_adj_OMC) # rescale function eq.(2)
            # combine eq.(2) and (3) in paper
            elif gen_mode == 'Complex':
                _adj_OMC = rescale_adj_matrix(_adj_OMC)  # rescale function eq.(2)
                _adj_OMC[_adj_OMC < t] = 0
                _adj_OMC[_adj_OMC >= t] = 1
                _adj_OMC = _adj_OMC * 0.25 / (_adj_OMC.sum(0, keepdims=True) + 1e-6)

            _adj_OMC = _adj_OMC / (_adj_OMC.sum(0, keepdims=True) + 1e-8)
            _adj_OMC = _adj_OMC + np.identity(num_classes, np.int)  # identity square matrix
            _A_adj['adj_O_M'] = _adj_OMC

        elif key == 'adj_all':
            _adj_all = result['adj_all']
            _adj_all = _adj_all / _nums
            # only eq.(3) in paper
            if gen_mode == 'ML_GCN':
                '''ML_GCN method'''
                _adj_all[_adj_all < t] = 0
                _adj_all[_adj_all >= t] = 1
                _adj_all = _adj_all * 0.25 / (_adj_all.sum(0, keepdims=True) + 1e-6)
            # only eq.(2) in paper
            elif gen_mode == 'My_rescale':
                '''Use My rescale function'''
                _adj_all = rescale_adj_matrix(_adj_all)  # rescale function eq.(2)
            # combine eq.(2) and (3) in paper
            elif gen_mode == 'Complex':
                _adj_all = rescale_adj_matrix(_adj_all)  # rescale function eq.(2)
                _adj_all[_adj_all < t] = 0
                _adj_all[_adj_all >= t] = 1
                _adj_all = _adj_all * 0.25 / (_adj_all.sum(0, keepdims=True) + 1e-6)
            _adj_all = _adj_all + np.identity(num_classes, np.int)  # identity square matrix
            _A_adj['adj_all'] = _adj_all

    return _A_adj


'''define the function to smooth the adj_matrix'''
def rescale_adj_matrix(adj_mat, t=5, p=0.02):
    """This function is to smooth the adj_matrix for dealing with the long-tail effect
    adj_mat: co-occurence adj matrix

    t: parameter_1, determine the amplify/shrink rate
    p: parameter_2, determine the borderline prob value of un-important concept to shrink
    context_word_length: we need to know the nums of context word,
                        because we need to suppress the role of context words for the whole representation
    """
    adj_mat_smooth = np.power(t, adj_mat - p) - np.power(t,  -p)
    return adj_mat_smooth


'''Laplacian Matrix transorm'''
def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj


'''Laplacian Matrix transform for concept graph'''
def gen_adj_concept(A):

    adj = {}
    for key, value in A.items():
        if key == 'adj_O_P':
            D = torch.pow(A['adj_O_P'].sum(1).float(), -0.5)
            D = torch.diag(D)
            adj['adj_O_P'] = torch.matmul(torch.matmul(A['adj_O_P'], D).t(), D)
            adj['adj_O_P'].detach()

        if key == 'adj_O_M':
            D = torch.pow(A['adj_O_M'].sum(1).float(), -0.5)
            D = torch.diag(D)
            adj['adj_O_M'] = torch.matmul(torch.matmul(A['adj_O_M'], D).t(), D)
            adj['adj_O_M'].detach()

        elif key == 'adj_all':
            D = torch.pow(A['adj_all'].sum(1).float(), -0.5)
            D = torch.diag(D)
            adj['adj_all'] = torch.matmul(torch.matmul(A['adj_all'], D).t(), D)
            adj['adj_all'].detach()

    return adj
