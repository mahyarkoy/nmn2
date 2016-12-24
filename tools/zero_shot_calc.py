# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 21:05:54 2016

@author: mahyarkoy
"""

import numpy as np
import json
from collections import defaultdict

jdata_path = '/media/evl/Public/Mahyar/Data/CVPRdata/results13/logs/test_predictions_10.json'
im_db = defaultdict(lambda: defaultdict(list))
ann_db = dict()
pred_db = dict()

def find_prediction(pred):
    top = 1    
    res = list()
    pred_list = pred.items()    
    for c, val in pred_list:
        vec = np.asarray(val)
        res.append(np.prod(vec))
    cid = np.argsort(res)[::-1][0:top]
    #cid = np.random.choice(len(res),top)
    out = [pred_list[x][0] for x in cid.tolist()]    
    return out, res

with open(jdata_path) as jf:
    jdata = json.load(jf)
    
for jdict in jdata:
    imn = jdict['im_name']
    imc = jdict['im_cid']
    sc = jdict['sent_cid']
    yes_pr = jdict['prob']
    im_db[imn][sc].append(yes_pr)
    ann_db[imn] = imc

acc = 0
for im, pred in im_db.items():
    preds, res = find_prediction(pred)
    pred_db[im] = preds    
    for p in preds:   
        if p == ann_db[im]:
            acc += 1
            continue

total = len(im_db.keys())
accuracy = acc / float(total)
print 'THIS IS HOW ACCURATE: '
print accuracy