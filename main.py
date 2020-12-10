from __future__ import division
import numpy as np
from model import *
from util import Data
import pickle
import argparse
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='yoochoose1_64', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--epoch', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--embSize', type=int, default=100, help='embedding size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--layer', type=float, default=2, help='the number of layer used')
parser.add_argument('--beta', type=float, default=0, help='ssl task maginitude')
opt = parser.parse_args()
train_data = pickle.load(open('../datasets/' + opt.dataset + '/train.txt', 'rb'))
test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))
if opt.dataset == 'diginetica':
    n_node = 43098
elif opt.dataset == 'yoochoose1_64_long' or opt.dataset == 'yoochoose1_4':
    n_node = 37484
else:
    n_node = 310
train_data = Data(train_data, n_node=n_node)
test_data = Data(test_data, n_node=n_node)
model = DHCN(emb_size=opt.embSize, batch_size=opt.batchSize, n_node=n_node, lr=opt.lr, l2=opt.l2, layers=opt.layer,beta=opt.beta)

print(opt)
top_K = [5,10,20]
best_results,metrics = {},{}
for K in top_K:
    best_results['epoch%d' %K] = [0,0]
    best_results['metric%d' %K] = [0,0]
for epoch in range(opt.epoch):
    print('epoch: ', epoch, '===========================================')
    slices = train_data.generate_batch(model.batch_size)
    fetches = [model.opt_rank, model.rank_loss, model.prediction, model.con_loss, model.item_embedding]
    print('start training: ', datetime.datetime.now())
    loss_ = []
    con = []
    for i, j in zip(slices, np.arange(len(slices))):
        items, tar, last, H, H_T, D, B, item_map, session_len, session_item = train_data.get_slice(i)
        A_hat, D_hat = train_data.get_overlap(items)
        _, loss, train_score, con_loss, item_emb = model.run(fetches, tar, items, last, H, H_T, D, B, item_map,
                                         session_len, session_item, A_hat, D_hat)
        con.append(con_loss)
        loss_.append(loss)
    loss = np.mean(loss_)
    con = np.mean(con)
    slices = test_data.generate_batch(model.batch_size)
    print('start predicting: ', datetime.datetime.now())
    test_loss = []
    for K in top_K:
        metrics['hit%d' % K] = []
        metrics['mrr%d' % K] = []
    for i, j in zip(slices, np.arange(len(slices))):
        items, tar, last, H, H_T, D, B, item_map, session_len, session_item = test_data.get_slice(i)
        A_hat, D_hat = test_data.get_overlap(items)
        scores, t_loss = model.run([model.prediction, model.rank_loss], tar, items, last,
                                      H, H_T, D, B, item_map, session_len, session_item,A_hat, D_hat)
        test_loss.append(t_loss)
        index = np.argsort(-scores, 1)  # descent order
        #calculating metrics
        for K in top_K:
            for prediction, target in zip(index[:, :K], tar):
                metrics['hit%d' %K].append(np.isin(target, prediction))
                if len(np.where(prediction == target)[0]) == 0:
                    metrics['mrr%d' %K].append(0)
                else:
                    metrics['mrr%d' %K].append(1 / (np.where(prediction == target)[0][0]+1))
    for K in top_K:
        metrics['hit%d' %K] = np.mean(metrics['hit%d' %K])*100
        metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
        if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
            best_results['metric%d' % K][0] = metrics['hit%d' % K]
            best_results['epoch%d' % K][0] = epoch
        if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
            best_results['metric%d' % K][1] = metrics['mrr%d' % K]
            best_results['epoch%d' % K][1] = epoch

    test_loss = np.mean(test_loss)
    for K in top_K:
        print('train_loss:\t%.4f\ttest_loss:\t%4f\tRecall@%d: %.4f\tMRR%d: %.4f\tEpoch: %d,  %d' %
          (loss, test_loss, K, best_results['metric%d' % K][0], K,best_results['metric%d' % K][1],
           best_results['epoch%d' % K][0], best_results['epoch%d' % K][1]))

