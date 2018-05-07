import numpy as np
import random
import tensorflow as tf
from sklearn.metrics import roc_auc_score

def assessment(abnormal_values, normal_values, seg_len, calc_mean = True):
    if calc_mean:
        abnormal_values = np.mean(np.split(abnormal_values,len(abnormal_values)//seg_len), axis = 1)
        normal_values = np.mean(np.split(normal_values,len(normal_values)//seg_len), axis = 1)
    else:
        abnormal_values = np.median(np.split(abnormal_values,len(abnormal_values)//seg_len), axis = 1)
        normal_values = np.median(np.split(normal_values,len(normal_values)//seg_len), axis = 1)
    labels = np.concatenate((np.ones(len(abnormal_values)),np.zeros(len(normal_values))), axis = 0)
    auc = roc_auc_score(labels, np.concatenate((abnormal_values, normal_values), axis=0))
    return auc

def assessment_full(prob_list_abnormal, prob_list_normal, seg_lens = [1, 120, 1200], title = ''):
    print(title)
    print('abnormal sample: %d, normal sample: %d' % (prob_list_abnormal.size, prob_list_normal.size))
    if isinstance(seg_lens, int):
        seg_lens = [seg_lens]
    results = np.zeros(len(seg_lens))
    for i in range(len(seg_lens)):
        auc = assessment(prob_list_abnormal, prob_list_normal, seg_lens[i], calc_mean = True)
        print("(length %4d) auc = %.3f" % (seg_lens[i], auc))
        results[i] = auc
    print('')
    return results

def shuffle_draw_batch(n_sample, batch_size):
    batch_indices = np.arange(n_sample)
    np.random.shuffle(batch_indices)
    n_batch = int(np.ceil(n_sample/batch_size))
    batch_list = []
    for i in range(n_batch):
        batch_list.append(batch_indices[i*batch_size:min((i+1)*batch_size, n_sample)])
    return batch_list
