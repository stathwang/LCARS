import os
import sys
import random
import argparse
import operator
import json
import time
import numpy as np
import pprint as pp
from collections import defaultdict

class ContentLDARec(object):

    def __init__(self, trainpath, testpath, ntopics, header=True):
        '''
        data: {UserID: [Item1, Item2, Item3, ...]} where duplicate items per user are allowed
        user2idx: {UserID: UserIndex}
        item2idx: {ItemID: ItemIndex}
        nusers: initialize total number of users
        nitems: initialize total number of items
        '''
        self.trainpath = trainpath
        self.testpath = testpath
        self.header = header
        self.ntopics = ntopics
        
        self.train_ui = defaultdict(list)       # training data: {UserID: [Item1, Item2, Item3, ...]} where duplicate items per user are allowed.
        self.train_uc = defaultdict(list)
        self.test_ui = defaultdict(list)        # test data: same format as self.train
        self.test_uc = defaultdict(list)
        
        self.user2idx = {}                      # user to index dict: {UserID: UserIndex}
        self.item2idx = {}                      # item to index dict: {ItemID: ItemIndex}
        self.cat2idx = {}                       # category (content word) to index dict: {CatID: CatIndex}
        self.idx2item = {}                      # index to item dict: {ItemIndex: ItemID}
        
        self.nusers = 0                         # initialize number of training users
        self.nitems = 0                         # initialize number of training items
        self.ncats = 0
        self.test_rowcount = 0

        self.recall = {}
        self.topk_user_item_scores = {}
        
        # Hyperparameters
        self.alpha = 50.0/ntopics
        self.beta = 0.01
        self.beta2 = 0.01
        self.burnin = 100
        self.thin = 1
        self.niters = 600
        self.topk_threshold = 40

    def read_data(self):
        '''
        Read the training dataset.
        Update data, user2idx, item2idx, nusers, nitems iteratively.
        '''
        with open(self.trainpath, 'r') as a, open(self.testpath, 'r') as b:
            if self.header:
                next(a)
                next(b)
            for line in a:
                line = line.strip('\n\r').split('\t')
                user = str(line[0])
                item = str(line[1])
                cat = str(line[3])
                
                if user not in self.train_ui:
                    self.user2idx[user] = self.nusers
                    self.nusers += 1

                if item not in self.item2idx:
                    self.item2idx[item] = self.nitems
                    self.idx2item[self.nitems] = item
                    self.nitems += 1

                if cat not in self.cat2idx:
                    self.cat2idx[cat] = self.ncats
                    self.ncats += 1

                self.train_ui[user].append(item)
                self.train_uc[user].append(cat)

            for line in b:
                self.test_rowcount += 1
                line = line.strip('\n\r').split('\t')
                user = str(line[0])
                item = str(line[1])
                cat = str(line[3])
                self.test_ui[user].append(item)
                self.test_uc[user].append(cat)

        a.close()
        b.close()

    def initialize_theta_phi(self):
        self.theta = np.zeros((self.nusers, self.ntopics))      # same dimension as UT
        self.phi = np.zeros((self.nitems, self.ntopics))        # same dimension as IT
        self.nu = np.zeros((self.ncats, self.ntopics))          # same dimension as CT

    def initialize_topics(self):
        '''
        Randomly initialize the parameter matrices.

        topic_assignment: {UserIndex: [TopicAssignment]} where the length of topic assignment is the number of spatial items visited by a user
        UT: a user x topic matrix
        IT: an item x topic matrix
        NT: a topic array
        NU: a user array
        '''
        if self.niters <= self.burnin:
            print('Burn-in should not exceed the number of iterations')
            sys.exit(0)

        print('Number of Users: ', self.nusers)
        print('Number of Topics: ', self.ntopics)
        print('Number of Items: ', self.nitems)
        print('Number of Content Words: ', self.ncats)

        self.topic_assignment = {}                                         # {user_index: [topic_assignment]}
        for user in self.train_ui:
            u = self.user2idx[user]
            self.topic_assignment[u] = [0 for item in self.train_ui[user]]
        
        self.NT = np.zeros(self.ntopics)
        self.NU = np.zeros(self.nusers)
        self.NC = np.zeros(self.ncats)
        self.IT = np.zeros((self.nitems, self.ntopics))
        self.UT = np.zeros((self.nusers, self.ntopics))
        self.CT = np.zeros((self.ncats, self.ntopics))
        self.IC = np.zeros((self.nitems, self.ncats))                      # item x category count matrix

        print('Randomly assigning topics')
        for user in self.train_ui:
            for j, (item, cat) in enumerate(zip(self.train_ui[user], self.train_uc[user])):
                u = self.user2idx[user]
                w = self.item2idx[item]
                c = self.cat2idx[cat]
                rt = random.randint(0, self.ntopics-1)
                self.topic_assignment[u][j] = rt
                self.IT[w, rt] += 1
                self.UT[u, rt] += 1
                self.CT[c, rt] += 1
                self.NT[rt] += 1
                self.NU[u] += 1
                self.NC[c] += 1
                self.IC[w, c] += 1

        # Initialize theta and phi matrices
        self.initialize_theta_phi()

    def update_topics(self, user, item, cat, item_pos, s):
        # Collapsed Gibbs sampling
        u = self.user2idx[user]
        w = self.item2idx[item]
        c = self.cat2idx[cat]
        z = self.topic_assignment[u][item_pos]
        self.IT[w, z] -= 1
        self.UT[u, z] -= 1
        self.CT[c, z] -= 1
        self.NT[z] -= 1
        self.NU[u] -= 1
        self.NC[c] -= 1

        multp = (self.UT[u,:] + self.alpha) / (np.sum(self.UT[u,:]) + self.ntopics * self.alpha) * \
                (self.IT[w,:] + self.beta) / (np.sum(self.IT, axis=0) + self.nitems * self.beta) * \
                (self.CT[c,:] + self.beta2) / (np.sum(self.CT, axis=0) + self.ncats * self.beta2)

        z_new = np.random.multinomial(1, multp/np.sum(multp)).argmax()
        self.topic_assignment[u][item_pos] = z_new
        self.IT[w, z_new] += 1
        self.UT[u, z_new] += 1
        self.CT[c, z_new] += 1
        self.NT[z_new] += 1
        self.NU[u] += 1
        self.NC[c] += 1

        self.compute_theta_phi(user, item, cat, item_pos, s)

    def compute_theta_phi(self, user, item, cat, item_pos, s):
        # Update theta, phi, and nu matrices only after burnin
        if s > self.burnin and s % self.thin == 0:
            inv_weight = self.thin / (s - self.burnin)
            u = self.user2idx[user]
            w = self.item2idx[item]
            c = self.cat2idx[cat]
            z_new = self.topic_assignment[u][item_pos]
            self.theta[u, z_new] += inv_weight * (self.UT[u, z_new] + self.alpha) / np.sum(self.UT[u,:] + self.alpha)
            self.phi[w, z_new] += inv_weight * (self.IT[w, z_new] + self.beta) / np.sum(self.IT[:, z_new] + self.beta) 
            self.nu[c, z_new] += inv_weight * (self.CT[c, z_new] + self.beta2) / np.sum(self.CT[:, z_new] + self.beta2)

    def recommend_top_k(self):
        pred = np.dot(self.theta, (self.phi * np.dot(self.IC, self.CT)).T)
        print(pred)
        print(pred.shape)
        for k in range(1, self.topk_threshold+1):
            print('Top-k threshold: ', k)
            hit = 0
            user_item_scores = {}
            for user in self.test_ui:
                #print('User:', user)
                u = self.user2idx[user]
                item_scores = {}
                for j, w in enumerate(pred[u,]):
                    item = self.idx2item[j]
                    item_scores[item] = float(w)
                temp = sorted(item_scores.items(), key=operator.itemgetter(1), reverse=True)
                top_n = [item for item, score in temp[0:k]]
                #print('Predicted Top-' + str(k) + ' Items:', ', '.join(top_n))
                #print('True Top-' + str(k) + ' Items:', ', '.join(self.test_ui[user][0:k]))
                #print('\n')
                #pp.pprint(item_scores)
                user_item_scores[user] = top_n
                for item in self.test_ui[user]:
                    if item in top_n:
                        hit += 1
            self.topk_user_item_scores[k] = user_item_scores
            self.recall[k] = float(hit)/self.test_rowcount
            
            print('Hit: ', hit)
            print('Rowcount: ', self.test_rowcount)
            print('Recall: ', float(hit)/self.test_rowcount)
            print('\n')

    def save_output(self):
        with open('recall_ca_lda.json', 'w') as rc:
            json.dump(self.recall, rc)
        with open('prediction_ca_lda.json', 'w') as pr:
            json.dump(self.topk_user_item_scores, pr)

    def main(self):
        self.read_data()
        self.initialize_topics()

        print('Collapsed Gibbs sampling in progress...')
        for s in range(1, self.niters+1):
            for user in self.train_ui:
                for item_pos, (item, cat) in enumerate(zip(self.train_ui[user], self.train_uc[user])):
                    self.update_topics(user, item, cat, item_pos, s)
        
        self.phi = self.phi / np.sum(self.phi, axis=0)                      # phi: item x topic matrix (a topic is a distribution over items)
        self.theta = (self.theta.T / np.sum(self.theta, axis=1)).T          # theta: user x topic matrix (a user is a distribution over topics)
        self.nu = self.nu / np.sum(self.nu, axis=0)                         # nu: cat x topic matrix (a topic is a distribution over content words)

        self.recommend_top_k()
        self.save_output()

if __name__=='__main__':

    filedir = 'data/'
    trainpath = filedir + 'douban_data_trunc_train.tsv'             # 75% in training set
    testpath = filedir + 'douban_data_trunc_test.tsv'               # 25% in test set
    
    tic = time.time()
    model = ContentLDARec(trainpath=trainpath, testpath=testpath, ntopics=150, header=True)
    model.main()
    toc = time.time()
    print('Model time: ' + str(1000*(toc-tic)) + ' ms')



