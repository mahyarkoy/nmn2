#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 21:55:43 2017
@author: mahyar

This script counts false positive, false negative,
and total occurances of each parse, in the given prediction file.
It also calculates the words occurances in each case, and generates output for d3.
The output shows total, false positive and false negative error of each word,
and also how much other words occumpanied that word in each case (contributed to error).
"""

import numpy as np
import json
from collections import defaultdict

jdata_path = '/media/evl/Public/Mahyar/Data/CVPRdata/results34/logs/train_predictions_5.json'
ignore_words = ['is', 'and']

### read validation prediction file
with open(jdata_path, 'r') as jf:
    jdata = json.load(jf)

'''
Count false positive and negative and total occurances of each parse
parse_db holds all parses as keys, and for each has word_list, false positive,
and false negative count and total count.
'''
words_count_p = defaultdict(int)
words_count_n = defaultdict(int)
parse_db = defaultdict(lambda: defaultdict(int))
for d in jdata:
    parse = d['parses']
    im_cid = d['im_cid']
    sent_cid = d['sent_cid']
    ann = 1 if d['answer']=='yes' else 0
    gt = 1 if im_cid == sent_cid else 0
    parse_db[parse]['count'] += 1
    if gt:
        parse_db[parse]['fn'] += 1 - ann
        parse_db[parse]['p'] += 1
    else:
        parse_db[parse]['fp'] += ann
        parse_db[parse]['n'] += 1
    ### making words dictionary
    parse_words_list = list()
    parse_words = parse.strip().replace('(', '').replace(')','').split()
    for pw in parse_words:
        if pw in ignore_words:
            continue
        words_count_p[pw] += gt
        words_count_n[pw] += 1 - gt
        parse_words_list.append(pw)
    parse_db[parse]['word_list'] = parse_words_list
                       
parse_list = parse_db.keys()
parse_errs = [(pdict['fp']+pdict['fn'])/float(pdict['count']) for pdict in parse_db.values()]
sorted_indices = np.argsort(parse_errs)
median_idx = sorted_indices[len(parse_errs)/2]
median_val = parse_errs[median_idx]
median_parse = parse_list[median_idx]
print 'MIN>>> ', parse_list[sorted_indices[0]], str(parse_errs[sorted_indices[0]])
print 'MEDIAN>>> ', median_parse, str(median_val)
print 'MAX>>> ', parse_list[sorted_indices[-1]], str(parse_errs[sorted_indices[-1]])

'''
Construct a list of filtered words, and a look up index dict. Also number of
occurances of each word when positive ground truth and when false ground truth.
'''
top_words = 80
word_list = list()
word_num_p = list()
word_num_n = list()
word_list = words_count_n.keys()
total_word_count = np.asarray(words_count_p.values()) + np.asarray(words_count_n.values())
sorted_words_idx = np.argsort(total_word_count)[::-1]
word_list = [word_list[idx] for idx in sorted_words_idx[:top_words]]
for w in word_list:
    word_num_p.append(words_count_p[w])
    word_num_n.append(words_count_n[w])
word_lookup = dict()
for idx, w in enumerate(word_list):
    word_lookup[w] = idx
               
'''
Filter out words from parse_db based on the total occurances of that words
'''
for p in parse_db.keys():
    wl = list()
    for w in parse_db[p]['word_list']:
        if w in word_list:
            wl.append(w)
    parse_db[p]['word_list'] = wl

'''
Create two matrix of cooccurances for false positive and false negative.
FP matrix shows error count on negative examples for each word on diag, and
contribution of all words to that error off diag.
FN matrix shows error count on positive examples for each word on diag, and 
contribution of all words to that error off diag.
'''
words_db = defaultdict(list)
freq_mat_fp = np.zeros((len(word_list), len(word_list)))
freq_mat_fn = np.zeros((len(word_list), len(word_list)))
freq_mat_total = np.zeros((len(word_list), len(word_list)))
freq_mat_p = np.zeros((len(word_list), len(word_list)))
freq_mat_n = np.zeros((len(word_list), len(word_list)))
for parse, freq in parse_db.items():
    parse_words = freq['word_list']
    for pw_i, pw in enumerate(parse_words):
        for pw2 in parse_words[pw_i:]:
            idx1 = word_lookup[pw]
            idx2 = word_lookup[pw2]
            freq_mat_fp[idx1, idx2] += freq['fp']
            freq_mat_fp[idx2, idx1] += freq['fp']
            freq_mat_fn[idx1, idx2] += freq['fn']
            freq_mat_fn[idx2, idx1] += freq['fn']
            freq_mat_total[idx1, idx2] += freq['count']
            freq_mat_total[idx2, idx1] += freq['count']
            freq_mat_p[idx1, idx2] += freq['p']
            freq_mat_p[idx2, idx1] += freq['p']
            freq_mat_n[idx1, idx2] += freq['n']
            freq_mat_n[idx2, idx1] += freq['n']
### check symmetry. Note that these wont be symmetric after normalization.
assert((freq_mat_fn.transpose() == freq_mat_fn).all())

'''
Normalize a given matrix:
1. divide diag elements by wnum vector (count of each word on each dim)
2. normalize off diag elements row wise
'''
def normalize_mat(mat, wnum):
    mat = np.copy(mat)
    shape = mat.shape
    mat_diag = np.diag(np.diag(mat)) * 1.0 / wnum
    mat[range(shape[0]), range(shape[1])] = 0
    vec_sum = np.sum(mat, axis=1) + 0.001
    vec_sum = vec_sum.reshape((shape[0],1))
    mat_norm = mat / vec_sum
    out_mat = mat_norm + mat_diag
    #assert((out_mat.transpose() == out_mat).all())
    return out_mat

'''
Normalize a given matrix:
1. divide diag elements by wnum vector (count of each word on each dim)
2. normalize off diag elements row wise

def normalize_mat(mat, wnum):
    mat = np.copy(mat)
    shape = mat.shape
    mat_diag = np.diag(np.diag(mat)) * 1.0 / wnum
    mat[range(shape[0]), range(shape[1])] = 0
    vec_sum = np.sum(mat, axis=1) + 0.001
    vec_sum = vec_sum.reshape((shape[0],1))
    mat_norm = mat / vec_sum
    out_mat = mat_norm + mat_diag
    #assert((out_mat.transpose() == out_mat).all())
    return out_mat
'''

'''
Normalize freq_mat s by the above normalization function.
FP_norm matrix shows error on negative examples for each word on diag, and
contribution of all words to that error off diag.
FN_norm matrix shows error on positive examples for each word on diag, and
contribution of all words to that error off diag.
'''
wnum_n = np.array(word_num_n)
wnum_p = np.array(word_num_p)
bias_denum = 0.000001

'''
Calculate error mat on total domain: positive and negative
'''
### N(X,F)
n_x_f = np.diag(freq_mat_fp) + np.diag(freq_mat_fn)
n_x_f = n_x_f.reshape(n_x_f.shape[0], 1)

### N(X)
n_x = np.diag(freq_mat_total)
n_x = n_x.reshape(n_x.shape[0],1)

### N(X,Y,F)
freq_mat = freq_mat_fp + freq_mat_fn

### P(Y|X,F) / P(Y|X) = (N(X,Y,F) / N(X,F) ) / ( N(X,Y) | N(X) )
effect_mat = (freq_mat / (n_x_f+bias_denum) + bias_denum) / (freq_mat_total / (n_x+bias_denum) + bias_denum)
### P(F|X,Y) = N(X,Y,F) / N(X,Y)
freq_mat_norm = freq_mat / (freq_mat_total + bias_denum)
effect_mat[range(effect_mat.shape[0]), range(effect_mat.shape[0])] = 0
effect_mat += np.diag(np.diag(freq_mat_norm))

'''
Calculate error mat on positive domain
'''
### N(X,F,P)
n_x_f_p = np.diag(freq_mat_fn)
n_x_f_p = n_x_f_p.reshape(n_x_f_p.shape[0], 1)

### N(X,P)
n_x_p = np.diag(freq_mat_p)
n_x_p = n_x_p.reshape(n_x_p.shape[0], 1)

### P(Y|X,F,P) / P(Y|X,P) = (N(X,Y,F,P) / N(X,F,P) ) / ( N(X,Y,P) | N(X,P) )
effect_mat_p = (freq_mat_fn / (n_x_f_p+bias_denum) + bias_denum) / (freq_mat_p / (n_x_p+bias_denum) + bias_denum)
### P(F|X,Y,P) = N(X,Y,F,P) / N(X,Y,P)
freq_mat_norm_p = freq_mat_fn / (freq_mat_p + bias_denum)
effect_mat_p[range(effect_mat_p.shape[0]), range(effect_mat_p.shape[0])] = 0
effect_mat_p += np.diag(np.diag(freq_mat_norm_p))

'''
Calculate error mat on negative domain
'''
### N(X,F,N)
n_x_f_n = np.diag(freq_mat_fp)
n_x_f_n = n_x_f_n.reshape(n_x_f_n.shape[0], 1)

### N(X,N)
n_x_n = np.diag(freq_mat_n)
n_x_n = n_x_n.reshape(n_x_n.shape[0], 1)

### P(Y|X,F,N) / P(Y|X,N) = (N(X,Y,F,N) / N(X,F,N) ) / ( N(X,Y,N) | N(X,N) )
effect_mat_n = (freq_mat_fp / (n_x_f_n+bias_denum) + bias_denum) / (freq_mat_n / (n_x_n+bias_denum) + bias_denum)
### P(F|X,Y,N) = N(X,Y,F,N) / N(X,Y,N)
freq_mat_norm_n = freq_mat_fp / (freq_mat_n + bias_denum)
effect_mat_n[range(effect_mat_n.shape[0]), range(effect_mat_n.shape[0])] = 0
effect_mat_n += np.diag(np.diag(freq_mat_norm_n))

'''
Calculate frequncy mat
'''
### P(Y|X) = N(X,Y) / N(X)
dist_mat = freq_mat_total / n_x
dist_diag = n_x / np.sum(n_x)
dist_mat[range(dist_mat.shape[0]), range(dist_mat.shape[0])] = 0
dist_mat += np.diag(dist_diag[:,0])

'''
### P(Y|X,F,N) = N(X,Y,F,N) / N(X,F,N) off diag
### P(F|X,N) = N(X,F,N) / N(X,N) on diag
freq_mat_fp_norm = normalize_mat(freq_mat_fp, wnum_n)

### P(Y|X,F,P) = N(X,Y,F,P) / N(X,F,P) off diag
### P(F|X,P) = N(X,F,P) / N(X,P) on diag
freq_mat_fn_norm = normalize_mat(freq_mat_fn, wnum_p)

### P(F|X,Y) = N(X,Y,F) / N(X,Y)
#freq_mat_norm = normalize_mat(freq_mat, wnum_n + wnum_p)
freq_mat_norm = freq_mat / (freq_mat_total + 0.00001)
'''
### Save as json to read by d3
jdata_list = list()
for idx, w in enumerate(word_list):
    freq_dict = dict()
    freq_dict['fp'] = effect_mat_n[idx,:].tolist()
    freq_dict['fn'] = effect_mat_p[idx,:].tolist()
    freq_dict['err'] = effect_mat[idx,:].tolist()
    freq_dict['dist'] = dist_mat[idx,:].tolist()
    jdata_list.append({'name': w, 'freq': freq_dict, 'id': idx, 'count': words_count_p[w]+words_count_n[w]})
    if idx == 9:
        f9 = freq_dict
        fx9 = freq_mat_fn[idx,:]
        w9 = w

with open('/home/mahyar/nmn_words_stat_train.json', 'w+') as jf:
    json.dump({'nodes': jdata_list},jf)

