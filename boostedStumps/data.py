#! /usr/bin/env python3
import random

import numpy as np


# Returns random data and random targets
def generate(n, num_features=2):
    return np.random.rand(n, num_features), np.random.choice([-1, 1], (n,))


def loadIris():
    out = []
    for l in open('../dat/iris.dat'):
        tup = l.strip('\n').split(',')

        if tup[-1] == '0':
            out.append((np.asarray([float(tup[0]), float(tup[1]), float(tup[2]), float(tup[3])]), 1))
        elif tup[-1] == '1':
            out.append((np.asarray([float(tup[0]), float(tup[1]), float(tup[2]), float(tup[3])]), -1))
    return out


def split(d):
    random.shuffle(d)
    s = int(.8 * len(d))
    train = d[:s]
    test = d[s:]

    trainX = np.vstack([l[0] for l in train])
    trainY = np.asarray([l[1] for l in train])
    testX = np.vstack([l[0] for l in test])
    testY = np.asarray([l[1] for l in test])

    return trainX, trainY, testX, testY
