#!/usr/bin/env python2

# check profiler
if not isinstance(__builtins__, dict) or "profile" not in __builtins__:
    __builtins__.__dict__["profile"] = lambda x: x

from misc import util
from misc.indices import QUESTION_INDEX, ANSWER_INDEX, MODULE_INDEX, MODULE_TYPE_INDEX, \
        NULL, NULL_ID, UNK_ID
from misc.visualizer import visualizer
import models
from models.nmn import MLPFindModule, MultiplicativeFindModule
import tasks
from collections import defaultdict
import apollocaffe
import argparse
import json
import logging.config
import random
import numpy as np
import yaml
from os import walk
import os

def main():
    ### Read config file and set config variables
    config = configure()

    ### Load indices if the load_indices is specified in config
    if hasattr(config.task, 'load_indices'):
        QUESTION_INDEX.load(config.task.load_indices+'question_index.json')
        MODULE_INDEX.load(config.task.load_indices+'module_index.json')
        ANSWER_INDEX.load(config.task.load_indices+'answer_index.json')
        #MODULE_TYPE_INDEX.load(config.model.load_indices+'module_type_index.json')
    
    ### Run main
    if config.real_time:
        interactive_main(config)
    else:
        auto_main(config)

def auto_main(config):
    task = tasks.load_task(config)
    model = models.build_model(config.model, config.opt)

    save_indices = config.task.save_indices if hasattr(config.task, 'save_indices') and config.task.save_indices else False
    save_net = config.task.save_net if hasattr(config.task, 'save_net') else 0
    i_epoch = 0
    logs = config.log_path
    while i_epoch <= config.opt.iters:
        ### Save the indices info only once
        if save_indices:
            QUESTION_INDEX.save(logs+'/question_index.json')
            MODULE_INDEX.save(logs+'/module_index.json')
            ANSWER_INDEX.save(logs+'/answer_index.json')
            save_indices = False
            #MODULE_TYPE_INDEX.save('logs/module_type_index.json')

        ### Load model if required, only once after the 0-th iteration
        if i_epoch == 0 and hasattr(config.model, 'load_model'):
            print('=====PRE-LOADING THE NET=====')
            train_path = config.task.load_train + '/itr_' + str(i_epoch)
            train_loss, train_acc, _ = \
                do_iter_external(train_path, task, model, config, train=False)
            model.load(config.model.load_model)
            if hasattr(config.model, 'load_adastate'):
                model.opt_state.load(config.model.load_adastate)
            i_epoch = 10 ### Set to what ever epoch should be continued

        print('=====TRAIN AT ITERATION %d=====' % i_epoch)            
        train_path = config.task.load_train + '/itr_' + str(i_epoch)
        if not os.path.isdir(train_path):
            print 'NO MORE BATCHES AVAILABLE AT ' + train_path
            break
        train_loss, train_acc, train_predictions = \
                do_iter_external(train_path, task, model, config, train=True)
        
        print('=====VALID AT ITERATION %d=====' % i_epoch) 
        val_loss, val_acc, val_predictions = \
                do_iter_external(config.task.load_val, task, model, config, vis='single')

        logging.info(
                "%5d  |  %8.3f  %8.3f  |  %8.3f  %8.3f",
                i_epoch,
                train_loss, val_loss,
                train_acc, val_acc)
        
        ### Save the net at each iteration
        if save_net > 0 and i_epoch%save_net == 0:
            model.save(logs+'/snapshots/model_%d.h5' % i_epoch)
            model.opt_state.save(logs+'/snapshots/model_%d_adastate' % i_epoch)

        ### Store Train prediction
        with open(logs+"/train_predictions_%d.json" % i_epoch, "w") as pred_f:
            print >>pred_f, json.dumps(train_predictions, indent=4)

        ### Store Validation prediction
        with open(logs+"/val_predictions_%d.json" % i_epoch, "w") as pred_f:
            print >>pred_f, json.dumps(val_predictions, indent=4)

        ### TEST RESULTS
        if i_epoch % 5 == 0 and i_epoch != 0:
            print('=====TEST AT ITERATION %d=====' % i_epoch)            
            test_loss, test_acc, test_predictions = \
                    do_iter_external(config.task.load_test, task, model, config)
            logging.info(
                    "TEST_%5d  |  %8.3f  |  %8.3f",
                    i_epoch,
                    test_loss, test_acc)
            with open(logs+"/test_predictions_%d.json" % i_epoch, "w") as pred_f:
                print >>pred_f, json.dumps(test_predictions)

        ### TEST_TRAIN RESULTS
        if False and i_epoch % 5 == 0 and i_epoch != 0 and hasattr(config.task, 'load_test_train'):
            print('=====TEST_TRAIN AT ITERATION %d=====' % i_epoch)            
            tt_loss, tt_acc, tt_predictions = \
                    do_iter_external(config.task.load_test_train, task, model, config)
            logging.info(
                    "TEST_TRAIN_%5d  |  %8.3f  |  %8.3f",
                    i_epoch,
                    tt_loss, tt_acc)
            with open(logs+"/test_train_predictions_%d.json" % i_epoch, "w") as pred_f:
                print >>pred_f, json.dumps(tt_predictions)
        
        i_epoch += 1

def data_from_file(fname):
    batch_list = list()
    with open(fname, 'r') as df:
        for l in df:
            cname, im_name, parse, ann = l.strip().split(';')
            question = parse.replace('(','').replace(')','')
            batch_list.append({'image':im_name+'.jpg.npz',
                               'parse':parse,
                               'question':question, 'answer': ann,
                               'cname':cname, 'cid':int(cname.split('.')[0]),
                               'sent_cid':0, 'sent_cname':'none'})
    return batch_list

def data_from_commandline():
    batch_list = list()
    while True:
        im_name = raw_input('Enter image name:').strip()
        if im_name == '_done':
            break
        cname = raw_input('Enter image class name:').strip()
        while True:
            parse = raw_input('Enter sps2 parse:').strip()
            if parse == '_done':
                break;
            ann = raw_input('Enter answer (yes/no):').strip()
            question = parse.replace('(','').replace(')','')
            batch_list.append({'image':im_name+'.jpg.npz',
                               'parse':parse,
                               'question':question, 'answer': ann,
                               'cname':cname, 'cid':int(cname.split('.')[0]),
                               'sent_cid':0, 'sent_cname':'none'})
    return batch_list

def setup_path(fpath):
    if os.path.exists(fpath):
        os.system('rm -r '+fpath)
    os.system('mkdir -p '+fpath)

def interactive_main(config):
    task = tasks.load_task(config)
    model = models.build_model(config.model, config.opt)

    save_indices = config.task.save_indices if hasattr(config.task, 'save_indices') else False
    save_net = config.task.save_net if hasattr(config.task, 'save_net') else 0
    i_epoch = 0
    batch_id = 0

    temp_data = 'temp_data'
    temp_log = 'temp_logs'
    config.log_path = temp_log
    setup_path(temp_data)
    setup_path(temp_log)

    load_model = raw_input('Enter path to model:')
    if config.data_file is None:
        batch_list = data_from_commandline()
    else:
        batch_list = data_from_file(config.data_file)
    
    with open(temp_data+'/batch_'+str(batch_id)+'.json', 'w+') as fj:
        json.dump(batch_list, fj, indent=4)

    print('=====PRE-LOADING THE NET=====')
    test_loss, test_acc, test_predictions = \
            do_iter_external(temp_data, task, model, config, train=False, vis=None)
    model.load(load_model)

    print('=====FORWARDING THE NET=====')
    test_loss, test_acc, test_predictions = \
            do_iter_external(temp_data, task, model, config, train=False, vis='all')
    
    print('=====RESULTS=====')
    print("TEST_%5d  |  %8.3f  |  %8.3f" % (i_epoch,test_loss,test_acc))
    with open(temp_log+"/test_predictions_%d.json" % i_epoch, "w") as pred_f:
        print >>pred_f, json.dumps(test_predictions)

def configure():
    apollocaffe.set_random_seed(0)
    np.random.seed(0)
    random.seed(0)
    apollocaffe.set_device(1)

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
            "-c", "--config", dest="config", required=True,
            help="model configuration file")
    arg_parser.add_argument(
            "-l", "--log-config", dest="log_config", default="config/log.yml",
            help="log configuration file")
    arg_parser.add_argument(
            "-lp", "--log-path", dest="log_path", default="logs",
            help="output log path.")
    arg_parser.add_argument(
            "-r", "--real-time", dest="real_time", default=False, action='store_true',
            help="real-time mode, uses data-file if specified.")
    arg_parser.add_argument(
            "-f", "--data-file", dest="data_file", default=None,
            help="data file to read batch_data.")

    args = arg_parser.parse_args()
    config_name = args.config.split("/")[-1].split(".")[0]

    with open(args.log_config) as log_config_f:
        log_filename = args.log_path+"/%s.log" % config_name
        log_config = yaml.load(log_config_f)
        log_config["handlers"]["fileHandler"]["filename"] = log_filename
        logging.config.dictConfig(log_config)

    with open(args.config) as config_f:
        config = util.Struct(**yaml.load(config_f))

    assert not hasattr(config, "name")
    config.name = config_name

    config.real_time = args.real_time
    config.data_file = args.data_file
    config.log_path = args.log_path

    return config

def do_iter(task_set, model, config, train=False, vis=False):
    loss = 0.0
    acc = 0.0
    predictions = []
    n_batches = 0

    # sort first to guarantee deterministic behavior with fixed seed
    data = list(sorted(task_set.data))
    np.random.shuffle(data)

    if vis:
        visualizer.begin(config.name, 100)

    for batch_start in range(0, len(data), config.opt.batch_size):
        batch_end = batch_start + config.opt.batch_size
        batch_data = data[batch_start:batch_end]

        batch_loss, batch_acc, batch_preds = do_batch(
                batch_data, model, config, train, vis)

        loss += batch_loss
        acc += batch_acc
        predictions += batch_preds
        n_batches += 1

        if vis:
            visualize(batch_data, model)

    if vis:
        visualizer.end()

    if n_batches == 0:
        return 0, 0, dict()
    assert len(predictions) == len(data)
    loss /= n_batches
    acc /= n_batches
    return loss, acc, predictions

def do_iter_external(pathname, task, model, config, train=False, vis=None):
    loss = 0.0
    acc = 0.0
    predictions = []
    n_batches = 0
    data_size = 0
    ### Read batches from file
    if vis:
        visualizer.begin(config.log_path+'/vis/'+pathname.split('/')[-1], 100)

    for (pname, dnames, fnames) in walk(pathname):
        for fn in fnames:
            print 'AT BATCH >>> ' + str(n_batches) + ' >>> ' + fn
            with open(pname+'/'+fn) as jf:
                jd = json.load(jf)
                batch_data_orig = task.read_batch_json(jd)

            #batch_data = batch_data_orig
            #imc = None
            batch_data, imc = reorder_batch(batch_data_orig)
            data_size += len(batch_data)
            batch_loss, batch_acc, batch_preds = do_batch(
                    batch_data, model, config, train, vis, im_count=imc)

            loss += batch_loss
            acc += batch_acc
            predictions += batch_preds
            n_batches += 1

            if vis == 'single':
                i_datum = np.random.choice(len(batch_data))
                datum = batch_data[i_datum]
                visualize(i_datum, datum, model)
            elif vis == 'all':
                for i_datum, datum in enumerate(batch_data):
                    visualize(i_datum, datum, model)

            if hasattr(config.task, 'debug'):
                if n_batches >= config.task.debug:
                    break
    if vis:
        visualizer.end()

    if n_batches == 0:
        return 0, 0, dict()
    assert len(predictions) == data_size
    loss /= n_batches
    acc /= n_batches
    return loss, acc, predictions

def reorder_batch(data):
    im_db = defaultdict(list)
    for idx, d in enumerate(data):
        im_db[d.im_name].append(idx)
    im_idx_mat = np.asarray(im_db.values())
    #print im_idx_mat.shape
    #print len(data)
    #print im_db.values()[0]
    new_data = [data[idx] for c in range(im_idx_mat.shape[-1]) for idx in im_idx_mat[:,c]]
    assert len(new_data) == len(data)
    return new_data, im_idx_mat.shape[0]


def do_batch(data, model, config, train, vis, im_count=None):
    predictions = forward(data, model, config, train, vis, im_count)
    answer_loss = backward(data, model, config, train, vis)
    acc = compute_acc(predictions, data, config)

    return answer_loss, acc, predictions

# TODO this is ugly and belongs somewhere else
def featurize_layouts(datum, max_layouts):
    # TODO pre-fill module type index
    layout_reprs = np.zeros((max_layouts, len(MODULE_INDEX) + 7))
    for i_layout in range(len(datum.layouts)):
        layout = datum.layouts[i_layout]
        labels = util.flatten(layout.labels)
        modules = util.flatten(layout.modules)
        for i_mod in range(len(modules)):
            if isinstance(modules[i_mod], MLPFindModule) or isinstance(modules[i_mod], MultiplicativeFindModule):
                layout_reprs[i_layout, labels[i_mod]] += 1
            mt = MODULE_TYPE_INDEX.index(modules[i_mod])
            layout_reprs[i_layout, len(MODULE_INDEX) + mt] += 1
    return layout_reprs

def forward(data, model, config, train, vis, im_count=None):
    model.reset(len(data))
    im_count = len(data) if im_count is None else im_count

    ### load batch data
    max_len = max(len(d.question) for d in data)
    max_layouts = max(len(d.layouts) for d in data)
    channels, height, width = data[0].load_features().shape
    #channels, size, trailing = data[0].load_features().shape
    #assert trailing == 1
    rel_features = None
    has_rel_features = data[0].load_rel_features() is not None
    questions = np.ones((len(data), max_len)) * NULL_ID
    #features = np.zeros((len(data), channels, height, width))
    features = np.zeros((im_count, channels, height, width))
    layout_reprs = np.zeros(
            (len(data), max_layouts, len(MODULE_INDEX) + 7))
    for i, datum in enumerate(data):
        questions[i, max_len-len(datum.question):] = datum.question
        if i < im_count:
            features[i, ...] = datum.load_features()
        ### uncomment for use in lstm
        #layout_reprs[i, ...] = featurize_layouts(datum, max_layouts)
    layouts = [d.layouts for d in data]

    ### apply model
    model.forward(
            layouts, layout_reprs, questions, features, rel_features, 
            dropout=(train and config.opt.dropout), deterministic=not train)

    ### extract predictions
    if config.opt.multiclass:
        pred_words = []
        for i in range(model.prediction_data.shape[0]):
            preds = model.prediction_data[i, :]
            chosen = np.where(preds > 0.5)[0]
            pred_words.append(set(ANSWER_INDEX.get(w) for w in chosen))
    else:
        pred_ids = np.argmax(model.prediction_data, axis=1)
        pred_words = [ANSWER_INDEX.get(w) for w in pred_ids]

    ### Store the top 10 scores
    top10_words = list()
    top10_probs = list()
    yes_prs = list()
    yes_id = ANSWER_INDEX.index('yes')
    for i in range(model.prediction_data.shape[0]):
        preds = model.prediction_data[i,:]
        chosen = list(reversed(np.argsort(preds)))
        top10_words.append(list(ANSWER_INDEX.get(w) for w in chosen))
        top10_probs.append(map(lambda x: '%.3f' %x, list(preds[w].item() for w in chosen)))
        yes_prs.append(preds[yes_id].item())

    ### Store predictions
    predictions = list()
    for i in range(len(data)):
        #qid = data[i].id
        answer = pred_words[i]
        top10 = dict(zip(top10_words[i],top10_probs[i]))
        predictions.append({'prob': yes_prs[i], 'im_name': data[i].im_name,
                            'im_cid': data[i].im_cid, 'im_cname':data[i].im_cname,
                            'sent_cid':data[i].sent_cid, 'sent_cname':data[i].sent_cname,
                            'answer': answer, 'parses': data[i].parses, 'top10': top10})

    return predictions

def backward(data, model, config, train, vis):
    n_answers = len(data[0].answers)
    loss = 0

    for i in range(n_answers):
        if config.opt.multiclass:
            output_i = np.zeros((len(data), len(ANSWER_INDEX)))
            for i_datum, datum in enumerate(data):
                for answer in datum.answers[i]:
                    output_i[i_datum, answer] = 1
        else:
            output_i = UNK_ID * np.ones(len(data))
            output_i[:len(data)] = \
                    np.asarray([d.answers[i] for d in data])
        loss += model.loss(output_i, multiclass=config.opt.multiclass)

    if train:
        model.train()

    return loss

def visualize(i_datum, datum, model):
    att_blobs = list()
    att_ids = list()
    mod_layout_choice = model.module_layout_choices[i_datum]
    ### FIXED: mod_index is not equal mod_layout_choice
    mod_index = model.nmns[datum.layouts[0].modules].index
    #print model.apollo_net.blobs.keys()
    for i in range(0,10):
        att_blob_name = "Find_%d_sigmoid" % (mod_index * 100 + i)
        if att_blob_name in model.apollo_net.blobs.keys():
            att_blobs.append(att_blob_name)
            att_ids.append('AT'+str(i))
    for i in range(0,10):
        att_blob_name = "And_%d_prod" % (mod_index * 100 + i)
        if att_blob_name in model.apollo_net.blobs.keys():
            att_blobs.append(att_blob_name)
            att_ids.append('AND'+str(i))
    ext_blob_ids = 'NONE'
    ext_val = -11
    for i in range(0,10):
        ext_blob_name = "Exists_%d_reduce" % (mod_index * 100 + i)
        if ext_blob_name in model.apollo_net.blobs.keys():
            ext_blob_ids='AT'+str(i)
            ext_val = model.apollo_net.blobs[ext_blob_name].data[i_datum,...].item()
            break
    if len(att_blobs) == 0:
        return

    #question = " ".join([QUESTION_INDEX.get(w) for w in datum.question[1:-1]]),
    preds = model.prediction_data[i_datum,:]
    top = np.argsort(preds)
    top_answers = list(reversed([ANSWER_INDEX.get(p) for p in top]))
    parse = datum.parses
    im_path = datum.input_image
    im_cid = datum.im_cid
    sent_cid = datum.sent_cid

    att_data_list = list()
    fields = list()
    fields.append("<img src='%s' width='140' height='140'/>" % im_path)
    for i_atb, atb in enumerate(att_blobs):
        att_data = model.apollo_net.blobs[atb].data[i_datum,...]
        att_data = att_data.reshape((14, 14))
        att_data_list.append(att_data)
        fields.append(att_data)
        fields.append(att_ids[i_atb])
    #att_data = np.zeros((14, 14))
    #chosen_parse = datum.parses[model.layout_ids[i_datum]]
    #fields.append(im_cid)
    fields.append(parse)
    fields.append(str(datum.layouts[0]))
    #fields.append(sent_cid)
    #fields.append(", ".join(top_answers))
    fields.append('TOP:'+top_answers[0])
    fields.append(", ".join([ANSWER_INDEX.get(a) for a in datum.answers]))
    fields.append('EXT_'+ext_blob_ids+'_%.3f'% ext_val)
    visualizer.show(fields)

def compute_acc(predictions, data, config):
    score = 0.0
    for prediction, datum in zip(predictions, data):
        pred_answer = prediction["answer"]
        if config.opt.multiclass:
            answers = [set(ANSWER_INDEX.get(aa) for aa in a) for a in datum.answers]
        else:
            answers = [ANSWER_INDEX.get(a) for a in datum.answers]

        matching_answers = [a for a in answers if a == pred_answer]
        if len(answers) == 1:
            score += len(matching_answers)
        else:
            score += min(len(matching_answers) / 3.0, 1.0)
    score /= len(data)
    return score

if __name__ == "__main__":
    main()

