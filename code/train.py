import time
import os
import pickle
import random
import shutil
import numpy as np
import sys
from tqdm import tqdm
import math
from DataInputWSE_es import DataInput
from tensorboardX import SummaryWriter
from model import *
from util_se import calculate_group_metric

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
BATCH_SIZE = 256
NEG_NUM = 5
REAL_BATCH_SIZE = BATCH_SIZE // (NEG_NUM+1)
LEARNING_RATE = 0.001
SEQ_LEN = 100
SE_NUM = 0 # don't care this parameter
DIMENSION = 16
NUM_PREFERENCE = 4
NUM_LAYER = 4
HIDDEN_SIZE = 16
DATASET = 'Toys_and_Games'#'Musical_Instruments'
EXP_NAME = 'YOUR_EXP_NAME'
DEVICE = 'cuda:0'


def get_exp_name(dataset, model_type, batch_size, lr, maxlen, save=True):
    extr_name = 'exp'#input('Please input the experiment name: ')
    para_name = '_'.join([dataset, model_type, 'b'+str(batch_size), 'lr'+str(lr), 'd'+str(DIMENSION), 'len'+str(maxlen)])
    exp_name = para_name + '_' + extr_name

    return exp_name


def main():
    random.seed(1234)
    np.random.seed(1234)

    # Hyper parameter
    USE_CKPT = False

    train_data = '../clear_data/{}_train_set.pkl'.format(DATASET)

    train_set = pickle.load(open(train_data, 'rb'))
    new_train_set = []
    last_time = 0
    for item in tqdm(train_set):
        assert last_time <= item[4]
        assert '1' == item[3]
        last_time = item[4]
        new_train_set.append(item)
    
    train_set_ = new_train_set
    del new_train_set

    all_records_len = len(train_set_)
    print('all records: {}'.format(all_records_len))
    len_1_5 = all_records_len // 5

    train_count = math.floor(all_records_len * 0.7)

    test_set = train_set[-len_1_5:]
    train_set = train_set[: train_count]

    user2id = pickle.load(open('../clear_data/{}_user2id.pkl'.format(DATASET), 'rb'))
    item2id = pickle.load(open('../clear_data/{}_item2id.pkl'.format(DATASET), 'rb'))

    user_count = len(user2id.keys())+1
    item_count = len(item2id.keys())+1
    print('user_count: {}, item_count: {}'.format(user_count, item_count))

    best_auc = 0.

    exp_name = get_exp_name(DATASET, "user2graph", BATCH_SIZE, LEARNING_RATE, SEQ_LEN)
    best_model_path = 'best_model/' + exp_name + '/'
    gpu_options = tf.GPUOptions(allow_growth=True)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        if USE_CKPT:
            model = modelTy(item_count, user_count, DIMENSION, 2002, SEQ_LEN, NEG_NUM, HIDDEN_SIZE, NUM_PREFERENCE, NUM_LAYER, SE_NUM)
            model.restore(sess, './best_model/CKPT_NAME/ckpt')
            test_iter = DataInput(test_set, 2, SEQ_LEN, 1000, item_count - 2, train_set, SE_NUM, init_es=False)    
            eval_user, eval_preds, eval_labels = [], [], []
            for bid, uij in tqdm(enumerate(test_iter), total=len(test_set)//2):
                user, tgt, label, hist, mask = uij
                logits = model.test(sess, [user, tgt, hist, mask])
                eval_user.extend(user)
                eval_preds.extend(logits)
                eval_labels.extend(label)
            metrics = calculate_group_metric(eval_labels, eval_preds, eval_user)
            for k in metrics:
                print("{}: {}".format(k, metrics[k])) 
            return
        else:
            model = modelTy(item_count, user_count, DIMENSION, BATCH_SIZE, SEQ_LEN, NEG_NUM, HIDDEN_SIZE, NUM_PREFERENCE, NUM_LAYER, SE_NUM)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

        exp_name = EXP_NAME + '_' + str(DATASET) + '_seq' + str(SEQ_LEN) + '_se' + str(SE_NUM) + '_lr' + str(LEARNING_RATE) + '_BATCH' + str(BATCH_SIZE)
        if not os.path.exists('./best_model/' + exp_name):
            os.mkdir('./best_model/' + exp_name)

        for e in range(50):
            random.shuffle(train_set)
            train_iter = DataInput(train_set, BATCH_SIZE // (NEG_NUM + 1), SEQ_LEN, NEG_NUM, item_count - 2, train_set, SE_NUM, init_es=False)
            test_iter = DataInput(test_set, BATCH_SIZE // (NEG_NUM + 1), SEQ_LEN, NEG_NUM, item_count - 2, train_set, SE_NUM, init_es=False)
            loss_sum = 0.
            epoch_time = 0.
            batch_num = 0
            for batch_idx, uij in tqdm(enumerate(train_iter), total=len(train_set)//REAL_BATCH_SIZE):
                user, tgt, label, hist, mask = uij
                
                start_time = int(time.time())
                loss = model.train(sess, [user, tgt, hist, mask, label, LEARNING_RATE])
                end_time = int(time.time())
                epoch_time += (end_time - start_time)
                batch_num += 1

                loss_sum += loss
            print("E Time: {}, AVG Batch Time: {}".format(epoch_time, epoch_time/batch_num))
            print("Epoch{}, loss: {}".format(e, loss_sum/(len(train_set)//REAL_BATCH_SIZE)))

            ## eval
            if 1:
                eval_user, eval_preds, eval_labels = [], [], []
                for bid, uij in tqdm(enumerate(test_iter), total=len(test_set)//REAL_BATCH_SIZE):
                    user, tgt, label, hist, mask = uij
                    logits = model.test(sess, [user, tgt, hist, mask])
                    eval_user.extend(user)
                    eval_preds.extend(logits)
                    eval_labels.extend(label)
                metrics = calculate_group_metric(eval_labels, eval_preds, eval_user)
                for k in metrics:
                    print("{}: {}".format(k, metrics[k]))
                if metrics['gauc'] > best_auc:
                    model.save(sess, './best_model/' + exp_name + '/' + 'ckpt')
                    best_auc = metrics['gauc']

if __name__ == '__main__':
    main()
