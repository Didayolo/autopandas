'''Compute the nearest neighbor adversarial accuracy'''
import os

'''
import sys
from itertools import product
import pickle as pkl
import concurrent.futures
import psutil
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
from tqdm import tqdm

class NearestNeighborMetrics():
    """ Calculate nearest neighbors and metrics.
    """

    def __init__(self, tr, te, synths):
        self.data = {'tr': tr, 'te': te}
        # add all synthetics
        for i, s in enumerate(synths):
            self.data['synth_{}'.format(i)] = s
        self.synth_keys = ['synth_{}'.format(i) for i in range(len(synths))]
        # pre allocate distances
        self.dists = {}

    def nearest_neighbors(self, t, s):
        """ Find nearest neighbors d_ts and d_ss.
        """
        # fit to S
        nn_s = NearestNeighbors(1).fit(self.data[s]) #.reshape(-1, 1)
        if t == s:
            # find distances from s to s (shortcut because it is itself)
            d = nn_s.kneighbors()[0]
        else:
            # find distances from t to s
            d = nn_s.kneighbors(self.data[t])[0]
        return t, s, d

    def compute_nn(self):
        """ Run all the nearest neighbors calculations.
        """
        # find all combinations of test, train, and synthetics
        tasks = product(self.data.keys(), repeat=2)

        # run multi-threaded
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            futures = [
                executor.submit(self.nearest_neighbors, t, s)
                for (t, s) in tasks
            ]
            # wait for each job to finish and output progress bar
            for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures)):
                # get result and store
                t, s, d = future.result()
                self.dists[(t, s)] = d

    def discrepancy_score(self, t, s):
        """ Calculate the NN discrepancy score.
        """
        left = np.mean(self.dists[(t, s)])
        right = np.mean(self.dists[(s, t)])
        return 0.5 * (left + right)

    def divergence(self, t, s):
        """ Calculate the NN divergence.
        """
        left = np.mean(np.log(self.dists[(t, s)] / self.dists[(t, t)]))
        right = np.mean(np.log(self.dists[(s, t)] / self.dists[(s, s)]))
        return 0.5 * (left + right)

    def adversarial_accuracy(self, t, s):
        """ Calculate the NN adversarial accuracy.
        """
        left = np.mean(self.dists[(t, s)] > self.dists[(t, t)])
        right = np.mean(self.dists[(s, t)] > self.dists[(s, s)])
        return 0.5 * (left + right)

    def compute_discrepancy(self):
        """ Compute the standard discrepancy scores.
        """
        # only one value
        j_rr = self.discrepancy_score('tr', 'te')
        j_ra = []
        j_rat = []
        j_aa = []
        # for all of the synthetic datasets to average
        for k in self.synth_keys:
            j_ra.append(self.discrepancy_score('tr', k))
            j_rat.append(self.discrepancy_score('te', k))
            # comparison to other synthetics
            for k_2 in self.synth_keys:
                if k != k_2:
                    j_aa.append(self.discrepancy_score(k, k_2))

        # average accross synthetics
        j_ra = np.mean(np.array(j_ra))
        j_rat = np.mean(np.array(j_rat))
        j_aa = np.mean(np.array(j_aa))
        return j_rr, j_ra, j_rat, j_aa

    def compute_divergence(self):
        """ Compute the standard divergence scores.
        """
        d_tr_a = []
        d_te_a = []
        for k in self.synth_keys:
            d_tr_a.append(self.divergence('tr', k))
            d_te_a.append(self.divergence('te', k))

        d_tr = np.mean(np.array(d_tr_a))
        d_te = np.mean(np.array(d_te_a))
        return d_tr, d_te

    def compute_adversarial_accuracy(self):
        """ Compute the standarad adversarial accuracy scores.
        """
        a_tr_a = []
        a_te_a = []
        for k in self.synth_keys:
            a_tr_a.append(self.adversarial_accuracy('tr', k))
            a_te_a.append(self.adversarial_accuracy('te', k))

        a_tr = np.mean(np.array(a_tr_a))
        a_te = np.mean(np.array(a_te_a))
        return a_tr, a_te
'''
