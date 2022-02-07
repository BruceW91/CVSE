from __future__ import print_function
import os
import pickle
import numpy
from collections import OrderedDict, Counter
import time
import numpy as np
from vocab import Vocabulary 
import torch
import copy

from util.utils import *
from data import get_test_loader



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        if self.count == 0:
            return str(self.val)
        return '%.4f (%.4f)' % (self.val, self.avg) 


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:    
            self.meters[k] = AverageMeter()     
        self.meters[k].update(v, n)     

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        # for i, (k, v) in enumerate(self.meters.iteritems()):   # python2
        for i, (k, v) in enumerate(self.meters.items()):   # python3
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        # for k, v in self.meters.iteritems():   # python2
        for k, v in self.meters.items():  # python3
            tb_logger.log_value(prefix + k, v.val, step=step)


def encode_data(model, data_loader, log_step=200, logging=print, alpha=None):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()
    # switch to evaluate mode
    model.val_start()
    end = time.time()

    # numpy array to keep all the embeddings
    img_embs_inst = None
    cap_embs_inst = None
    img_embs_cons = None
    cap_embs_cons = None
    concept_labels = None

    for i, (images, captions, attribute_labels, attribute_input_embs, lengths, ids) in enumerate(data_loader):
        model.logger = val_logger
        img_emb, cap_emb, predict_score_v, predict_score_t = model.forward_emb(images, captions,
                                                                               attribute_labels, attribute_input_embs,
                                                                               lengths, alpha, volatile=True)
        # Only get the fused features
        img_emb_inst = img_emb[2]
        cap_emb_inst = cap_emb[2]   
        img_emb_cons = img_emb[1]
        cap_emb_cons = cap_emb[1]  

        # initialize the numpy arrays given the size of the embeddings
        if cap_embs_inst is None:
            img_embs_inst = np.zeros((len(data_loader.dataset), img_emb_inst.size(1))) 
            cap_embs_inst =  np.zeros((len(data_loader.dataset), cap_emb_inst.size(1))) 
        if cap_embs_cons is None:
            img_embs_cons = np.zeros((len(data_loader.dataset), img_emb_cons.size(1)))   
            cap_embs_cons = np.zeros((len(data_loader.dataset), cap_emb_cons.size(1)))   
        if concept_labels is None:
            concept_labels = np.zeros((len(data_loader.dataset), attribute_labels.size(1)))   

        # preserve the embeddings by copying from gpu and converting to numpy
        img_embs_inst[ids] = img_emb_inst.data.cpu().numpy().copy()
        cap_embs_inst[ids] = cap_emb_inst.data.cpu().numpy().copy()
        img_embs_cons[ids] = img_emb_cons.data.cpu().numpy().copy()
        cap_embs_cons[ids] = cap_emb_cons.data.cpu().numpy().copy()
        concept_labels[ids] = attribute_labels.data.cpu().numpy().copy()

        # measure accuracy and record loss
        model.forward_loss(img_emb, cap_emb, predict_score_v, predict_score_t, model.dataset_name)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logging('Test: [{0}/{1}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    .format(
                        i, len(data_loader), batch_time=batch_time, 
                        e_log=str(model.logger)))   
        del images, captions  

    return img_embs_inst, cap_embs_inst, img_embs_cons, cap_embs_cons, concept_labels


def encode_data_KNN_rerank(model, data_loader, log_step=200, logging=print, index_KNN_neighbour=None, concept_labels=None, alpha=None):
    """
    Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()
    # switch to evaluate mode
    model.val_start()
    end = time.time()

    # numpy array to keep all the embeddings
    img_embs_inst = None
    cap_embs_inst = None
    img_embs_cons = None
    cap_embs_cons = None
    Complete_labels_all = None

    if not isinstance(index_KNN_neighbour,list):
        index_KNN_neighbour = index_KNN_neighbour.astype(int)

    for i, (images, captions, attribute_labels, attribute_input_embs, lengths, ids) in enumerate(data_loader):
        model.logger = val_logger
        Complete_labels = copy.deepcopy(attribute_labels); Complete_labels = Complete_labels.numpy()  # convert to array

        # complete the concept label
        for j in range(attribute_labels.shape[0]):
            neighhbour_index = index_KNN_neighbour[ids[j]]
            neighhbour_index.append(ids[j])
            K_neighbour_labels = concept_labels[neighhbour_index]
            Complete_labels[j] = K_neighbour_labels.max(axis=0)     # get the initial extended concept labels

        Complete_labels = torch.from_numpy(Complete_labels)
        img_emb, cap_emb, predict_score_v, predict_score_t = model.forward_emb(images, captions,
                                                                               Complete_labels, 
                                                                               attribute_input_embs,
                                                                               lengths, alpha, volatile=True)
        # Only get the fused features
        img_emb_inst = img_emb[2]
        cap_emb_inst = cap_emb[2]       
        img_emb_cons = img_emb[1]
        cap_emb_cons = cap_emb[1]

        # initialize the numpy arrays given the size of the embeddings
        if cap_embs_inst is None:
            img_embs_inst = np.zeros((len(data_loader.dataset), img_emb_inst.size(1)))
            cap_embs_inst =  np.zeros((len(data_loader.dataset), cap_emb_inst.size(1)))
        if cap_embs_cons is None:
            img_embs_cons = np.zeros((len(data_loader.dataset), img_emb_cons.size(1)))
            cap_embs_cons = np.zeros((len(data_loader.dataset), cap_emb_cons.size(1)))
        if Complete_labels_all is None:
            Complete_labels_all = np.zeros((len(data_loader.dataset), Complete_labels.size(1)))

        # preserve the embeddings by copying from gpu and converting to numpy
        img_embs_inst[ids] = img_emb_inst.data.cpu().numpy().copy()
        cap_embs_inst[ids] = cap_emb_inst.data.cpu().numpy().copy()
        img_embs_cons[ids] = img_emb_cons.data.cpu().numpy().copy()
        cap_embs_cons[ids] = cap_emb_cons.data.cpu().numpy().copy()
        Complete_labels_all[ids] = Complete_labels.data.cpu().numpy().copy()

        # measure accuracy and record loss
        model.forward_loss(img_emb, cap_emb, predict_score_v, predict_score_t, model.dataset_name)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logging('Test: [{0}/{1}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    .format(
                        i, len(data_loader), batch_time=batch_time, 
                        e_log=str(model.logger)))   
        del images, captions  

    return img_embs_inst, cap_embs_inst, img_embs_cons, cap_embs_cons, Complete_labels_all


def evalrank(model_path, data_path=None, data_name=None, data_name_vocab=None, split='dev', fold5=False,
            VSE_model=None, data_loader=None, concept_path=None, transfer_test=False, concept_name=None):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    start_epoch = checkpoint['epoch']
    best_rsum = checkpoint['best_rsum']
    print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
          .format(opt.resume, start_epoch, best_rsum))

    if data_path is not None:
        opt.data_path = data_path
    if data_name is not None:   
        opt.data_name = data_name   

    # Jugde whether to use transfering testing results
    if transfer_test == True:
        opt.attribute_path = concept_path
    if concept_name is not None:
        opt.concept_name = concept_name
    if 'coco' in opt.data_name:
        fuse_weight = 0.9   
    elif 'f30k' in opt.data_name:   
        fuse_weight = 0.85  

    print(opt)
    print("=> loading checkpoint '{}'".format(opt.resume))

    with open(os.path.join(opt.vocab_path,
                           '%s_vocab.pkl' % data_name_vocab), 'rb') as f:   
        vocab = pickle.load(f)

    opt.vocab_size = len(vocab)
    word2idx = vocab.word2idx

    # construct model
    model = VSE_model(word2idx, opt)  # if with channel attention
    # load model state
    model.load_state_dict(checkpoint['model'])

    print('Loading dataset')
    data_loader = get_test_loader(split, opt.data_name, vocab,
                                  opt.batch_size, opt.workers, transfer_test, opt)      
    print('Computing results...')
    img_embs, cap_embs, img_emb_cons, cap_emb_cons, concept_labels = encode_data(model=model, data_loader=data_loader, alpha=fuse_weight)

    '''2). Make label completation'''   
    ind_cap_complete = label_complete(concept_label=concept_labels, img_embs=img_embs, cap_embs=cap_embs, data_name=opt.data_name)

    img_embs, cap_embs, img_emb_cons, cap_emb_cons, completion_labels = encode_data_KNN_rerank(model=model, data_loader=data_loader,
                                                                                             index_KNN_neighbour=ind_cap_complete, concept_labels=concept_labels,
                                                                                             alpha=fuse_weight)
    print('Images: %d, Captions: %d' %
          (img_embs.shape[0] // 5, cap_embs.shape[0]), " for testing")

    if not fold5:
        # no cross-validation, full evaluation
        r, rt = i2t_sep_sim(img_embs, cap_embs, img_emb_cons, cap_emb_cons, opt.data_name,   
                            weight_fused=0.95,
                            measure=opt.measure, return_ranks=True)
        ri, rti = t2i_sep_sim(img_embs, cap_embs, img_emb_cons, cap_emb_cons, opt.data_name,   
                            weight_fused=0.95,
                            measure=opt.measure, return_ranks=True)
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3       
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        print("rsum: %.1f" % rsum)
        print("Average i2t Recall: %.1f" % ar)
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
        print("Average t2i Recall: %.1f" % ari)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)

    else:
        # 5fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            r, rt0 = i2t_sep_sim(img_embs[i * 5000:(i + 1) * 5000], cap_embs[i * 5000:(i + 1) * 5000],
                                img_emb_cons[i * 5000:(i + 1) * 5000], cap_emb_cons[i * 5000:(i + 1) * 5000],
                                opt.data_name, 
                                weight_fused=0.95, measure=opt.measure, return_ranks=True) 
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i_sep_sim(img_embs[i * 5000:(i + 1) * 5000],  cap_embs[i * 5000:(i + 1) * 5000],
                           img_emb_cons[i * 5000:(i + 1) * 5000], cap_emb_cons[i * 5000:(i + 1) * 5000],
                           opt.data_name, 
                           weight_fused=0.95, measure=opt.measure, return_ranks=True)
            if i == 0:  
                rt, rti = rt0, rti0
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)   

            ar = (r[0] + r[1] + r[2]) / 3   
            ari = (ri[0] + ri[1] + ri[2]) / 3   
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]   
            print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))        
            results += [list(r) + list(ri) + [ar, ari, rsum]]           

        print("-----------------------------------")    
        print("Mean metrics: ") 
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())  
        print("rsum: %.1f" % (mean_metrics[10] * 6))    
        print("Average i2t Recall: %.1f" % mean_metrics[11])    
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" %   
              mean_metrics[:5]) 
        print("Average t2i Recall: %.1f" % mean_metrics[12])
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[5:10])

    torch.save({'rt': rt, 'rti': rti}, 'ranks.pth.tar')


def i2t_sep_sim(images, captions, img_emb_cons, cap_emb_cons, data_name=None, weight_fused=0.8, npts=None, measure='cosine', return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = images.shape[0] // 5
    index_list = []
    ranks = numpy.zeros(npts)
    top1 = numpy.zeros(npts)
    for index in range(npts):
        # Get query image
        im = images[5 * index].reshape(1, images.shape[1])
        im_cons = img_emb_cons[5 * index].reshape(1, images.shape[1])
        # Compute scores
        if measure == 'cosine':
            d_inst = numpy.dot(im, captions.T).flatten()
            d_cons = numpy.dot(im_cons, cap_emb_cons.T).flatten()
        # weighted sum of two level similarities
        d = weight_fused * d_inst + (1-weight_fused) * d_cons
        inds = numpy.argsort(d)[::-1]
        index_list.append(inds[0])      

        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = numpy.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp  
        ranks[index] = rank 
        top1[index] = inds[0]   

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i_sep_sim(images, captions, img_emb_cons, cap_emb_cons, data_name=None, weight_fused=0.8, npts=None, measure='cosine', return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """ 
    # if 'coco' in data_name or 'coco' == data_name:
    #     weight_fused -= 0.05

    if npts is None:
        npts = images.shape[0] // 5
    ims_inst = numpy.array([images[i] for i in range(0, len(images), 5)])
    ims_cons = numpy.array([img_emb_cons[i] for i in range(0, len(img_emb_cons), 5)]) 
    ranks = numpy.zeros(5 * npts)
    top1 = numpy.zeros(5 * npts)
    for index in range(npts):
        # Get query captions
        queries_inst = captions[5 * index:5 * index + 5]
        queries_cons = cap_emb_cons[5 * index:5 * index + 5]
        # Compute scores
        if measure == 'cosine':
            d_inst = numpy.dot(queries_inst, ims_inst.T)
            d_cons = numpy.dot(queries_cons, ims_cons.T)
         # weighted sum of two level similarities
        d = weight_fused * d_inst + (1-weight_fused) * d_cons
        inds = numpy.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = numpy.argsort(d[i])[::-1]
            ranks[5 * index + i] = numpy.where(inds[i] == index)[0][0]
            top1[5 * index + i] = inds[i][0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:    
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def calculate_sim(img_emb, cap_emb):
    mat_sim = numpy.dot(img_emb, cap_emb.T)
    image_num = img_emb.shape[0] // 5
    mat_sim_2 = np.zeros((image_num, image_num*5)) # (N(image), 5N(caption))
    for index in range(image_num):
        sim_line = mat_sim[index*5]
        mat_sim_2[index] = sim_line
    return mat_sim_2

def construct_N_with_KNN(mat_sim_inst, K_neigs=3, return_index=True):

    order_t2i = np.argsort(-mat_sim_inst, 0)
    order_i2t = np.argsort(-mat_sim_inst, 1)
    index_N_neighbours = np.zeros((mat_sim_inst.shape[1], K_neigs))
    interval = int(K_neigs/2)

    for i in range(mat_sim_inst.shape[1]):  
        ind_nearest_t2i = order_t2i[0, i]
        cand_nearest_i2t = order_i2t[ind_nearest_t2i]
        index_i_txt = np.where(cand_nearest_i2t == i); index_i_txt = index_i_txt[0][0]
        if index_i_txt < cand_nearest_i2t.shape[0] - K_neigs:
            if K_neigs == 3 and index_i_txt>1:  
                index_N_neighbours[i] = cand_nearest_i2t[index_i_txt-1:index_i_txt+2]   
            # elif K_neigs != 3 and K_neigs % 2 == 0:
            elif K_neigs > 3 and K_neigs % 2 == 0:
                index_N_neighbours[i] = cand_nearest_i2t[index_i_txt-interval:index_i_txt+interval]    
            elif K_neigs > 3 and K_neigs % 2 != 0:
                index_N_neighbours[i] = cand_nearest_i2t[index_i_txt-interval-1:index_i_txt+interval]   
            else:
                index_N_neighbours[i] = cand_nearest_i2t[:K_neigs] 
        else:
            index_N_neighbours[i] = cand_nearest_i2t[-K_neigs:] 
    return index_N_neighbours

def label_complete(concept_label, img_embs, cap_embs, data_name):
    cap_embs = torch.Tensor(cap_embs)       
    # txt emb KNN search
    print('Computing K-nearest Neighbours for sentences...')    
    index_K_neighbour_t2t = construct_H_with_KNN(cap_embs, K_neigs=10, return_index=True)  
    print('Finish t2t K-nearest Neighbours Computing.')

    cap_embs = cap_embs.numpy()
    # use img-to-txt sim to find nearest Neighbours for sentences
    mat_sim_inst = calculate_sim(img_embs, cap_embs)        
    index_K_neighbour_i2t = construct_N_with_KNN(mat_sim_inst, K_neigs=3, return_index=True)
    print('Finish i2t K-nearest Neighbours Computing.')
    ind_cap_complete = []

    for i in range(len(index_K_neighbour_t2t)):
        a = index_K_neighbour_t2t[i];  a = [int(elem) for elem in a]
        b = index_K_neighbour_i2t[i];  b = [int(elem) for elem in b] 
        tmp = [val for val in a if val in b]
        if len(tmp) < 3 or tmp == []:
            if 'f30k' in data_name or 'f30k' == data_name:
                tmp = a[:3] + b[:4]
            elif 'coco' in data_name or 'coco' == data_name:
                tmp = a[:2] + b[:1]   
                # tmp = a[:2] + b[:2]
        ind_cap_complete.append(tmp)

    return ind_cap_complete









