import os
import sys
import random
import argparse
import operator
import json
import time
import numpy as np
import pprint as pp

class LDARec(object):

    def __init__(self, trainpath, testpath, ntopics, header=True):
        self.trainpath = trainpath
        self.testpath = testpath
        self.header = header
        self.ntopics = ntopics
        self.train = {}                  # training data: {UserID: [Item1, Item2, Item3, ...]} where duplicate items per user are allowed.
        self.test = {}                   # test data: same format as self.train
        self.user2idx = {}               # user to index dict: {UserID: UserIndex}
        self.item2idx = {}               # item to index dict: {ItemID: ItemIndex}
        self.idx2item = {}               # index to item dict: {ItemIndex: ItemID}
        self.nusers = 0                  # initialize number of training users
        self.nitems = 0                  # initialize number of training items
        self.test_rowcount = 0
        
        self.recall = {}
        self.topk_user_item_scores = {}
        
        # Hyperparameters
        self.alpha = 50.0/ntopics
        self.beta = 0.01
        self.burnin = 100
        self.thin = 1
        self.niters = 600
        self.topk_threshold = 40

    def read_data(self):
        '''
        Load the training dataset and read in data values iteratively.
        '''
        with open(self.trainpath, 'r') as a, open(self.testpath, 'r') as b:
            if self.header:
                next(a)
                next(b)
            for line in a:
                line = line.strip('\n\r').split('\t')
                user = str(line[0])
                item = str(line[1])
                if user not in self.train:
                    self.train[user] = [item]
                    self.user2idx[user] = self.nusers
                    self.nusers += 1
                else:
                    self.train[user].append(item)
                if item not in self.item2idx:
                    self.item2idx[item] = self.nitems
                    self.idx2item[self.nitems] = item
                    self.nitems += 1
            for line in b:
                self.test_rowcount += 1
                line = line.strip('\n\r').split('\t')
                user = str(line[0])
                item = str(line[1])
                if user not in self.test:
                    self.test[user] = [item]
                else:
                    self.test[user].append(item)

        a.close()
        b.close()

        # Randomly shuffle items within each user
        for user in self.train:
            random.shuffle(self.train[user])

    def initialize_theta_phi(self):
        self.theta = np.zeros((self.nusers, self.ntopics))      # same dimension as UT
        self.phi = np.zeros((self.nitems, self.ntopics))        # same dimension as VT

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

        self.topic_assignment = {}                                         # {user_index: [topic_assignment]}
        for user in self.train:
            u = self.user2idx[user]
            self.topic_assignment[u] = [0 for item in self.train[user]]
        self.UT = np.zeros((self.nusers, self.ntopics))
        self.IT = np.zeros((self.nitems, self.ntopics))
        self.NT = np.zeros(self.ntopics)
        self.NU = np.zeros(self.nusers)

        print('Randomly assigning topics')
        for user in self.train:
            for j, item in enumerate(self.train[user]):
                u = self.user2idx[user]
                w = self.item2idx[item]
                rt = random.randint(0, self.ntopics-1)
                self.topic_assignment[u][j] = rt
                self.IT[w, rt] += 1
                self.UT[u, rt] += 1
                self.NT[rt] += 1
                self.NU[u] += 1

        # Initialize theta and phi matrices
        self.initialize_theta_phi()

    def update_topics(self, user, item, item_pos, s):
        # Collapsed Gibbs sampling
        u = self.user2idx[user]
        w = self.item2idx[item]
        z = self.topic_assignment[u][item_pos]
        self.IT[w, z] -= 1
        self.UT[u, z] -= 1
        self.NT[z] -= 1
        self.NU[u] -= 1

        multp = (self.UT[u,:] + self.alpha) * (self.IT[w,:] + self.beta) / (self.NT + self.nitems * self.beta) 
        z_new = np.random.multinomial(1, multp/np.sum(multp)).argmax()
        self.topic_assignment[u][item_pos] = z_new
        self.IT[w, z_new] += 1
        self.UT[u, z_new] += 1
        self.NT[z_new] += 1
        self.NU[u] += 1

        self.compute_theta_phi(user, item, item_pos, s)

    def compute_theta_phi(self, user, item, item_pos, s):
        # Update theta and phi matrices only after burnin
        if s > self.burnin and s % self.thin == 0:
            inv_weight = self.thin / (s - self.burnin)
            u = self.user2idx[user]
            w = self.item2idx[item]
            z_new = self.topic_assignment[u][item_pos]
            self.theta[u, z_new] += inv_weight * (self.UT[u, z_new] + self.alpha) / np.sum(self.UT[u,:] + self.alpha)
            self.phi[w, z_new] += inv_weight * (self.IT[w, z_new] + self.beta) / np.sum(self.IT[:, z_new] + self.beta) 

    def recommend_top_k(self):
        pred = np.dot(self.theta, self.phi.T)                       # user x item prediction matrix
        for k in range(1, self.topk_threshold+1):
            print('Top-k threshold: ', k)
            hit = 0
            user_item_scores = {}
            for user in self.test:
                #print('User:', user)
                u = self.user2idx[user]
                item_scores = {}
                for j, w in enumerate(pred[u,]):
                    item = self.idx2item[j]
                    item_scores[item] = float(w)
                temp = sorted(item_scores.items(), key=operator.itemgetter(1), reverse=True)
                top_n = [item for item, score in temp[0:k]]
                #print('Predicted Top-' + str(k) + ' Items:', ', '.join(top_n))
                #print('True Top-' + str(k) + ' Items:', ', '.join(self.test[user][0:k]))
                #print('\n')
                #pp.pprint(item_scores)
                user_item_scores[user] = top_n
                for item in self.test[user]:
                    if item in top_n:
                        hit += 1
            self.topk_user_item_scores[k] = user_item_scores
            self.recall[k] = float(hit)/self.test_rowcount
            
            print('Hit: ', hit)
            print('Rowcount: ', self.test_rowcount)
            print('Recall: ', float(hit)/self.test_rowcount)
            print('\n')

    def save_output(self):
        with open('recall.json', 'w') as rc:
            json.dump(self.recall, rc)
        with open('prediction.json', 'w') as pr:
            json.dump(self.topk_user_item_scores, pr)

    def main(self):
        self.read_data()
        self.initialize_topics()
        
        print('Collapsed Gibbs sampling in progress...')
        for s in range(1, self.niters+1):
            for user in self.train:
                for item_pos, item in enumerate(self.train[user]):
                    self.update_topics(user, item, item_pos, s)
        
        self.phi = self.phi / np.sum(self.phi, axis=0)                      # phi: item x topic matrix (a topic is a distribution over items)
        self.theta = (self.theta.T / np.sum(self.theta, axis=1)).T          # theta: user x topic matrix (a user is a distribution over topics)

        self.recommend_top_k()
        self.save_output()

if __name__=='__main__':
    
    filedir = '/data/'
    trainpath = filedir + 'douban_data_trunc_train.tsv'             # 75% in training set
    testpath = filedir + 'douban_data_trunc_test.tsv'               # 25% in test set
    
    tic = time.time()
    model = LDARec(trainpath=trainpath, testpath=testpath, ntopics=150, header=True)
    model.main()
    toc = time.time()
    print('Model time: ' + str(1000*(toc-tic)) + ' ms')
