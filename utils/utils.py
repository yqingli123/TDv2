import random
import numpy as np
from torch import nn
from .latex2gtd_v2_2 import list2node, node2list_shuffle

# load dictionary
def load_dict(dictFile):
    lexicon = {}
    with open(dictFile) as fp:
        stuff = fp.readlines()
        for l in stuff:
            w = l.strip().split()
            lexicon[w[0]] = int(w[1])   
    return lexicon

# create batch
def prepare_data(params, images_x, seqs_y, seqs_key, object2id, relation2id, shuffle=False):
    heights_x = [s.shape[1] for s in images_x]
    widths_x = [s.shape[2] for s in images_x]
    lengths_y = [len(s) for s in seqs_y]

    n_samples = len(heights_x)
    max_height_x = np.max(heights_x)
    max_width_x = np.max(widths_x)
    maxlen_y = np.max(lengths_y)
    num_relation = len(relation2id)

    x = np.zeros((n_samples, params['input_channels'], max_height_x, max_width_x)).astype(np.float32) - 1
    childs    = np.zeros((maxlen_y, n_samples)).astype(np.int64)  
    parents   = np.zeros((maxlen_y, n_samples)).astype(np.int64)
    c_pos     = np.zeros((maxlen_y, n_samples)).astype(np.int64)
    p_pos     = np.zeros((maxlen_y, n_samples)).astype(np.int64)
    relations = np.zeros((maxlen_y, n_samples)).astype(np.int64)
    pathes    = np.zeros((maxlen_y, n_samples, num_relation)).astype(np.float32)

    x_mask = np.zeros((n_samples, max_height_x, max_width_x)).astype(np.float32)
    y_mask = np.zeros((maxlen_y, n_samples)).astype(np.float32)

    for idx, (s_x, s_y, s_key) in enumerate(zip(images_x, seqs_y, seqs_key)):
        x[idx, :, :heights_x[idx], :widths_x[idx]] = (255 - s_x) / 255.
        x_mask[idx, :heights_x[idx], :widths_x[idx]] = 1.
        if shuffle:
            tree = list2node(s_y)
            s_y = []
            node2list_shuffle('<s>', 1, 'Start', tree, s_y, True)
        for i, line in enumerate(s_y):
            c_id, p_id, re_id = object2id[line[0]], object2id[line[2]], relation2id[line[4]]
            childs[i, idx]    = c_id
            parents[i, idx]   = p_id
            c_pos[i, idx]     = int(line[1])
            p_pos[i, idx]     = int(line[3])
            relations[i, idx] = re_id
            pathes[i, idx, -1] = 1
            if i > 0:
                pathes[int(line[3])-1, idx, -1] = 0
                pathes[int(line[3])-1, idx, re_id] = 1

        y_mask[:lengths_y[idx], idx] = 1.

    return x, x_mask, childs, y_mask, parents, relations, pathes, c_pos, p_pos


# init model params
def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        try:
            nn.init.constant_(m.bias.data, 0.)
        except:
            pass

    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        try:
            nn.init.constant_(m.bias.data, 0.)
        except:
            pass

# compute metric
def cmp_result(rec,label):
    dist_mat = np.zeros((len(label)+1, len(rec)+1),dtype='int32')
    dist_mat[0,:] = range(len(rec) + 1)
    dist_mat[:,0] = range(len(label) + 1)
    for i in range(1, len(label) + 1):
        for j in range(1, len(rec) + 1):
            hit_score = dist_mat[i-1, j-1] + (label[i-1] != rec[j-1])
            ins_score = dist_mat[i,j-1] + 1
            del_score = dist_mat[i-1, j] + 1
            dist_mat[i,j] = min(hit_score, ins_score, del_score)

    dist = dist_mat[len(label), len(rec)]
    return dist, len(label)




