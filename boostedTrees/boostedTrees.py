#! /usr/bin/env python3
import math

import data
import numpy as np


class BoostedTrees(object):
    def __init__(self, n=20, n_features=3, rand=False):
        self.r = rand
        if rand:
            self.n = n
            self.d, self.t = data.generate(n, num_features=n_features)
            assert len(self.d) == len(self.t)
        else:
            # d = data.loadIris()
            # self.d, self.t, self.testd, self.testt = data.split(d)
            # self.n = len(self.d)

            # self.d, self.t = data.loadCredit()
            self.d, self.t, self.testd, self.testt = data.loadCredit()
            self.n = len(self.d)

        print("Data shape: {}".format(self.d.shape))

        self.stumps = self.makeFewerStumps()

        print("Tree space: {}".format(len(self.stumps)))

        self.weights = np.ones((self.n,)) * float(1 / self.n)
        self.alphas = []
        self.H = []
        self.acc = []

    def generate_model(self):
        i = 0
        while self.test(self.d, self.t) < 1 and i < 3:
            print("Iteration {}".format(i+1))

            try:
                stump = self.next_stump()
            except ValueError:
                print("Out of stumps! Using best model...")
                idx = self.acc.index(max(self.acc))
                print("Best model is iteration {} with accuracy {}".format(idx, self.acc[idx]))
                self.H = self.H[:min(idx+1,len(self.acc))]
                break

            error = self.get_error(stump)
            print("Error: {}".format(error))
            self.H.append(stump)
            self.stumps.remove(stump)

            try:
                self.alphas.append((.5 * math.log((1 - error) / error)))
            except ZeroDivisionError:
                self.alphas.append(1)
                print("Perfect rule found!")
                break

            acc = self.test(self.d, self.t)
            print("Training set accuracy: {}".format(acc))
            self.acc.append(acc)

            self.reweight(stump)
            i+=1

        if self.r == False:
            print("Testing on independent set")
            print("Testing set accuracy: {}".format(self.test(self.testd, self.testt)))

    # Implements Thank God Hole No. 2
    def reweight(self, stump):
        error = self.get_error(stump)
        correctZ = 1 / ((1 - error) * 2)
        incorrectZ = 1 / (error * 2)

        for i in range(self.n):
            if stump(self.d[i]) != self.t[i]:
                self.weights[i] *= incorrectZ
            else:
                self.weights[i] *= correctZ

    def get_error(self, stump):
        s = 0
        for i in range(self.n):
            if stump(self.d[i]) != self.t[i]:
                s += self.weights[i]
        return s

    def next_stump(self):
        return min(self.stumps, key=self.get_error)

    def test(self, x, y):
        if len(self.H) <= 0:
            return 0
        else:
            n = len(x)
            preds = [self.predict(x[i]) for i in range(n)]
            correct = sum([1 if preds[i] == y[i] else 0 for i in range(n)])
            return correct / n

    def predict(self, x):
        raw = sum([self.alphas[i] * self.H[i](x) for i in range(len(self.H))])
        return 1 if raw >= 0 else -1

    def makeRule(self, direct, target, dim, v):
        if direct == 0:
            def func(x):
                if x[dim] < v:
                    return target
                else:
                    return -1 * target

            return func
        else:
            def func(x):
                if x[dim] >= v:
                    return target
                else:
                    return -1 * target

            return func

    def makeStumps(self):
        rules = []
        for dim in range(self.d.shape[1]):
            # For now make all the rules, then reduce later
            rules.append(self.makeRule(1, 1, dim, self.d[0, dim]))
            rules.append(self.makeRule(1, -1, dim, self.d[0, dim]))

            # counter = 0
            for i in range(self.d.shape[0] - 1):
                r1 = self.makeRule(0, 1, dim, self.d[i, dim])
                r2 = self.makeRule(0, -1, dim, self.d[i, dim])

                rules.append(r1)
                rules.append(r2)

        return rules

    # Uses Thank God Hole No. 2
    # Doesn't provide the same results. Not entirely sure why
    def makeFewerStumps(self):
        rules = []
        for dim in range(self.d.shape[1]):
            attrs = [(self.d[i, dim], self.t[i]) for i in range(self.n)]

            rules.append(self.makeRule(0, 1, dim, attrs[0][0]))
            rules.append(self.makeRule(0, -1, dim, attrs[0][0]))
            curr = None

            i = 0
            while i < self.n - 1:
                # print(i)
                while attrs[i][1] == curr and i < self.n - 1:
                    i += 1
                rules.append(self.makeRule(1, 1, dim, attrs[i][0]))
                rules.append(self.makeRule(1, -1, dim, attrs[i][0]))
                curr = attrs[i][1]
                i += 1

        return rules


if __name__ == '__main__':
    btm = BoostedTrees(rand=False, n=500, n_features=5)
    btm.generate_model()
