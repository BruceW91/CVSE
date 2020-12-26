# -----------------------------------------------------------
# Consensus-Aware Visual-Semantic Embedding implementation based on
# "VSE++: Improving Visual-Semantic Embeddings with Hard Negatives"
# "Consensus-Aware Visual-Semantic Embedding for Image-Text Matching"
# Haoran Wang, Ying Zhang, Zhong Ji, Yanwei Pang, Lin Ma
#
# Writen by Haoran Wang, 2020
# ---------------------------------------------------------------


import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
from PIL import Image
import numpy as np
import json as jsonmod
from pycocotools.coco import COCO
import pickle

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res



'''1) MSCOCO dataset'''
class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """
    def __init__(self, cap_json_path, data_path, attribute_path, data_split, vocab, opt):
        '''Use to coco object to load file_name of images'''
        if isinstance(cap_json_path, tuple):
            self.coco = (COCO(cap_json_path[0]), COCO(cap_json_path[1]))  # load coco data from json file with COCO protocol provided by importing pycocotools.coco
            self.coco_restval = self.coco[0]
            '''merge the image_ids for using restval COCO train set'''
            self.coco_restval.imgs = Merge(self.coco_restval.imgs, self.coco[1].imgs)
        else:
            self.coco = COCO(cap_json_path)
            self.coco_restval = self.coco

        self.vocab = vocab

        loc = data_path + '/'

        # Captions
        self.captions = []
        with open(loc+'%s_caps.txt' % data_split, 'rb') as f:
            for line in f:
                self.captions.append(line.strip())

        # Image features
        print ("Image path", loc + '%s_ims.npy' % data_split)
        self.images = np.load(loc+'%s_ims.npy' % data_split)
        self.length = len(self.captions)
        print("Len in captions", self.length)

        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000
        
        self.data_split = data_split

        # load the image ids for loading the original images
        self.image_ids = []
        img_file_name = loc + '%s_ids.txt' % data_split
        with open(img_file_name, 'rb') as f:
            for line in f:
                line = int(line)
                self.image_ids.append(line)

        print("Original images in data_loader", len(self.image_ids))

        self.num_classes = opt.num_attribute

        # load coco concept_annotation
        self.attribute_json_dir = attribute_path['attribute']
        self.attribute_name_json_dir = attribute_path['attribute_name']

        '''load the concrete words of concepts'''
        with open(opt.concept_name, "r") as names_concepts:
            name_concepts_coco = jsonmod.load(names_concepts)
        self.name_concepts = []
        for i, (k,v) in enumerate(name_concepts_coco.items()):
            k = lemmatizer.lemmatize(k)            # get the lemma form of words
            self.name_concepts.append(k)    

        self.get_anno()
        self.num_classes = len(self.cat2idx)        

        # load the intial glove word embedding file of concepts
        with open(opt.inp_name, 'rb') as f:
            self.attribue_input_emb = pickle.load(f)        

    # Load coco concept json file
    def get_anno(self):

        if isinstance(self.attribute_json_dir, tuple):
            list_1_path = self.attribute_json_dir[0]
            list_2_path = self.attribute_json_dir[1]
            self.img_list = ( jsonmod.load(open(list_1_path, 'r')), jsonmod.load(open(list_2_path, 'r')) )
            self.cat2idx = jsonmod.load(open(self.attribute_name_json_dir, 'r'))
        else:
            list_path = self.attribute_json_dir
            self.img_list = jsonmod.load(open(list_path, 'r'))
            self.cat2idx = jsonmod.load(open(self.attribute_name_json_dir, 'r'))

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index // self.im_div
        image = torch.Tensor(self.images[img_id])
        caption = self.captions[index]
        vocab = self.vocab      

        sent = str(caption.strip())
        sent = sent.lstrip('b')  # remove the beginning mark

        # Convert caption (string) to word ids.
        tokens_sent = nltk.tokenize.word_tokenize(
            sent.lower())   
        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        
        if self.data_split == 'train':

            # Load the file name of original images
            image_id = self.image_ids[img_id]
            coco = self.coco_restval
            img_file_name = coco.loadImgs(image_id)[0]['file_name']
            
            # a) load the annoted labels 
            attribute_label = np.ones(self.num_classes,
                                       np.float32) / self.num_classes
            for img_att_pair in self.img_list:
                if img_att_pair['file_name'] == img_file_name:
                    attribute_label = self.get(img_att_pair)
                    break
            attribute_label = torch.Tensor(attribute_label)

        else:
            '''b) generate the real label for each sentence'''
            attribute_label = np.zeros(self.num_classes, np.float32)                
            for (i, word) in enumerate(tokens_sent):            
                try:        
                    word_lemma = lemmatizer.lemmatize(word)                         
                    # word_lemma = word     
                except:         
                    continue        
                if word_lemma in self.name_concepts:            
                    inx_concept = self.name_concepts.index(word_lemma)                  
                    attribute_label[inx_concept] = 1            

            attribute_label = torch.Tensor(attribute_label)     

        # load the input word embeddings for concepts   
        attri_input_emb = self.attribue_input_emb; attri_input_emb = torch.Tensor(attri_input_emb) 

        return image, target, attribute_label, attri_input_emb, index, img_id
        
    def get(self, item):
        # load concept labels
        labels = sorted(item['concept_labels'])
        target = np.zeros(self.num_classes, np.float32)
        target[labels] = 1
        return target

    def __len__(self):
        return self.length



'''2) Flickr30k dataset'''
class PrecompDataset_Flickr30k(data.Dataset):
    """
    Load precomputed captions and image features for f30k dataset
    """
    def __init__(self, data_path, attribute_path, data_split, vocab, opt):
        self.vocab = vocab
        loc = data_path + '/'

        # 1) Captions
        self.captions = []
        with open(loc+'%s_caps.txt' % data_split, 'rb') as f:
            for line in f:
                self.captions.append(line.strip())

        # 2) Image features
        self.images = np.load(loc + '%s_ims.npy' % data_split)
        self.length = len(self.captions)
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        if data_split == 'dev' or data_split == 'test':
            self.length = 5000

        self.data_split = data_split

        '''Load the image ids for loading the corresponding concept labels'''
        self.image_ids = []
        img_file_name = loc + '%s_ids.txt' % data_split
        with open(img_file_name, 'rb') as f:
            for line in f:
                line = int(line)
                self.image_ids.append(line) 

        '''load Flickr30k concept annotation'''
        self.attribute_json_dir = attribute_path        
        
        '''load the concrete words of concepts'''
        with open(opt.concept_name, "r") as names_concepts:
            name_concepts_f30k = jsonmod.load(names_concepts)
        self.name_concepts = []
        for i, (k,v) in enumerate(name_concepts_f30k.items()):
            k = lemmatizer.lemmatize(k)            # get the lemma form of words
            self.name_concepts.append(k)    

        self.img_list = jsonmod.load(open(self.attribute_json_dir, 'r'))
        self.num_classes = opt.num_attribute

        '''load the intial glove word embedding file of concepts'''
        with open(opt.inp_name, 'rb') as f:
            self.attribue_input_emb = pickle.load(f)

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index // self.im_div
        image = torch.Tensor(self.images[img_id])
        caption = self.captions[index]
        vocab = self.vocab

        sent = str(caption.strip())
        sent = sent.lstrip('b')  # remove the beginning mark

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower())   
        tokens_sent = nltk.tokenize.word_tokenize(
            sent.lower())   
        
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])  
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)  

        # Load the concept labels
        image_id = self.image_ids[img_id]   
        image_id = str(image_id)    # convert to string     

        if self.data_split == 'train':
            attribute_label = np.ones(self.num_classes,
                                    np.float32) / self.num_classes  # change for avoiding the empty labels
            # a) load the annoted labels 
            for img_att in self.img_list:     
                if img_att['img_id'] == image_id:
                    attribute_label = self.get(img_att)   
                    break     
            attribute_label = torch.Tensor(attribute_label)     
            
        else:
            '''b) generate the real label for each sentence'''
            attribute_label = np.zeros(self.num_classes, np.float32)                
            for (i, word) in enumerate(tokens_sent):            
                try:        
                    word_lemma = lemmatizer.lemmatize(word)                         
                    # word_lemma = word     
                except:         
                    continue        
                if word_lemma in self.name_concepts:            
                    inx_concept = self.name_concepts.index(word_lemma)                  
                    attribute_label[inx_concept] = 1            

            attribute_label = torch.Tensor(attribute_label)     

        # load the input word embeddings for concepts           
        attri_input_emb = self.attribue_input_emb; attri_input_emb = torch.Tensor(attri_input_emb)      

        return image, target, attribute_label, attri_input_emb, index, img_id       

    def get(self, item):    
        '''load concept labels'''       
        labels = sorted(item['concept_labels'])         
        target = np.zeros(self.num_classes, np.float32)     
        target[labels] = 1
        return target   

    def __len__(self):  
        return self.length          



def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption, concept_label, concept_emb) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
            - attribute_label: concept label, torch tensor of shape (concept_num);
            - attribute_input_emb: initial concept embeddings, torch tensor of shape (concept_num, word_emb_dim);
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        attribute_label: torch tensor of shape (concept_num);
        attribute_input_emb: torch tensor of shape (concept_num, word_emb_dim);
        lengths: list; valid length for each padded caption.
        ids: index
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)

    images, captions, attribute_label, attribute_input_emb, ids, img_ids = zip(*data)
    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)
    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    attribute_labels = torch.stack(attribute_label, 0)
    attribute_input_embs = torch.stack(attribute_input_emb, 0)

    return images, targets, attribute_labels, attribute_input_embs, lengths, ids


def get_precomp_loader(orig_cap_path, data_path, attribute_path, data_split, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=20, drop_last=True):
    if 'coco' in data_path:
        dset = PrecompDataset(orig_cap_path, data_path,
                              attribute_path, data_split, vocab, opt)

    elif 'f30k' in data_path:
        dset = PrecompDataset_Flickr30k(data_path,
                              attribute_path, data_split, vocab, opt)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn,
                                              drop_last=drop_last)
    return data_loader


def get_loaders(data_name, vocab, batch_size, workers, opt):

    dpath = os.path.join(opt.data_path, data_name)

    # concept file path
    orig_dpath = os.path.join(opt.orig_img_path, opt.orig_data_name)

    orig_path, attribute_path = get_paths(orig_dpath, opt.attribute_path, opt.data_name)
    train_loader = get_precomp_loader(orig_path['train']['cap'], dpath, attribute_path, 'train', vocab, opt,
                                      batch_size, True, workers, drop_last=True)
    val_loader = get_precomp_loader(orig_path['val']['cap'], dpath, attribute_path, 'dev', vocab, opt,
                                    batch_size, False, workers, drop_last=False)
    return train_loader, val_loader


def get_test_loader(split_name, data_name, vocab, batch_size,
                    workers, transfer_test, opt):
    dpath = os.path.join(opt.data_path, data_name)

    orig_dpath = os.path.join(opt.orig_img_path, opt.orig_data_name)

    orig_path, attribute_path = get_paths(orig_dpath, opt.attribute_path, opt.data_name, transfer_test)
    test_loader = get_precomp_loader(orig_path['val']['cap'], dpath, attribute_path, split_name, vocab, opt,
                                    batch_size, False, workers, drop_last=False)

    return test_loader



def get_paths(path, attribute_path, name='coco', transfer_test=False, use_restval=True):
    """
    Returns paths to images and annotations for the given datasets. For MSCOCO
    indices are also returned to control the data split being used.
    The indices are extracted from the Karpathy et al. splits using this
    snippet:

    >>> import json
    >>> dataset=json.load(open('dataset_coco.json','r'))
    >>> A=[]
    >>> for i in range(len(D['images'])):
    ...   if D['images'][i]['split'] == 'val':
    ...     A+=D['images'][i]['sentids'][:5]
    ...
    :param name: Dataset names
    :param use_restval: If True, the the `restval` data is included in train.
    """
    roots = {}
    ids = {}
    if 'coco_precomp' in name:
        imgdir = os.path.join(path, 'images')
        capdir = os.path.join(path, 'annotations')
        roots['train'] = {
            'img': os.path.join(imgdir, 'train2014'),
            'cap': os.path.join(capdir, 'captions_train2014.json')
        }
        roots['val'] = {
            'img': os.path.join(imgdir, 'val2014'),
            'cap': os.path.join(capdir, 'captions_val2014.json')
        }
        roots['test'] = {
            'img': os.path.join(imgdir, 'val2014'),
            'cap': os.path.join(capdir, 'captions_val2014.json')
        }

        roots['trainrestval'] = {
            'img': os.path.join(imgdir, 'trainrestval2014'),
            'cap': (roots['train']['cap'], roots['val']['cap'])
        }
        if use_restval:
            roots['train'] = roots['trainrestval']

        roots_anno = {}
        if use_restval:
            roots_anno = {
                'attribute': os.path.join(attribute_path, 'trainval_concept_label.json'),
                'attribute_name': ( os.path.join(attribute_path, 'category_concepts.json') )
            }
        else:
            roots_anno = {
                'attribute':  os.path.join(attribute_path, 'train_anno.json'),
                'attribute_name': os.path.join(attribute_path, 'category.json')
            }

    elif 'f30k_precomp' in name:
        imgdir = os.path.join(path, 'images')
        cap = os.path.join(path, 'dataset_flickr30k.json')
        roots['train'] = {'img': imgdir, 'cap': cap}
        roots['val'] = {'img': imgdir, 'cap': cap}
        roots['test'] = {'img': imgdir, 'cap': cap}

        # load the file path of concept labels
        if transfer_test == False:
            roots_anno = os.path.join(attribute_path, 'all_f30k_concept_label.json')
        else:
            roots_anno = os.path.join(attribute_path, 'Flickr30k_test_concept_label.json')

    return roots, roots_anno



