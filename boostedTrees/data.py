#! /usr/bin/env python3
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# Returns random data and random targets
def generate(n, num_features=2):
    return np.random.rand(n, num_features), np.random.choice([-1, 1], (n,))


def loadCredit():
    df = pd.read_csv('../dat/crx.dat')
    df.dropna(how='all')
    df = pd.get_dummies(data=df, columns=['A', 'D', 'E', 'F', 'G', 'I', 'J', 'L', 'M'])

    train, test = train_test_split(df, test_size=0.2)
    trainy = train['P']
    testy = test['P']

    trainx = train.drop('P', axis=1)
    testx = test.drop('P', axis=1)

    # y = df['P']
    # x = df.drop('P', axis=1)
    # return x.as_matrix(), y.as_matrix()

    return trainx.as_matrix(), trainy.as_matrix(), testx.as_matrix(), testy.as_matrix()


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


if __name__ == '__main__':
    d = loadCredit()
