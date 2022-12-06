from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

import numpy as np
import pandas as pd
import scipy.io as sio
import random

from LDL import bfgs_ldl
from metrics import score

class ssg_ldl:

    def __init__(self, s, n=300, k=5, fx=0.5, fy=0.5):
        self.n = n
        self.k = k
        self.fx = fx
        self.fy = fy

        self.X, self.y = s['X'], s['y']
        self.dist = fx * np.sum(np.linalg.norm(np.repeat([self.X], self.X.shape[0], axis=0)
            .transpose(1, 0, 2) - self.X, axis=2), axis=1) / self.X.shape[0] + \
            fy * np.sum(np.linalg.norm(np.repeat([self.y], self.y.shape[0], axis=0)
            .transpose(1, 0, 2) - self.y, axis=2), axis=1) / self.y.shape[0]

        self.knn = NearestNeighbors(n_neighbors=self.k)
        self.knn.fit(np.concatenate([self.X, self.y], axis=1))

        self.X_ = self.X
        self.y_ = self.y

    def select_sample(self):
        total_dist = np.sum(self.dist)
        r = random.random() * total_dist
        for i in range(self.X.shape[0]):
            r -= self.dist[i]
            if r < 0:
                return i

    def create_synthetic_sample(self, i):
        _, index = self.knn.kneighbors(np.concatenate([self.X[i], self.y[i]]).reshape(1, -1))
        nnarray = index.reshape(-1)
        nn = random.randint(1, self.k)

        dif = self.X[nnarray[nn-1]] - self.X[i]
        gap = np.random.random(self.X[0].shape)

        X = self.X[i] + gap * dif
        y = np.average(self.y[nnarray], axis=0)

        self.X_ = np.concatenate([self.X_, X.reshape(1, -1)])
        self.y_ = np.concatenate([self.y_, y.reshape(1, -1)])

    def fit(self):
        t = self.X.shape[0] * self.n // 100

        for _ in range(t):
            self.create_synthetic_sample(self.select_sample())

        return {'X': self.X_, 'y': self.y_}


if __name__ == '__main__':

    seed = 114514
    random.seed(seed)
    np.random.seed(seed)

    data = sio.loadmat('dataset/' + 'SJAFFE')

    X, y = data['X'], data['y']

    model = ssg_ldl(data)
    data_o = model.fit()

    X_o, y_o = data_o['X'], data_o['y']

    columns = ["Chebyshev", "Clark", "Canberra",
        "K-L", "Cosine", "Intersection", "Euclidean", "Sorensen", "Squared-Chi2", "Fidelity"]

    df1 = pd.DataFrame(columns=columns)
    df2 = pd.DataFrame(columns=columns)

    for i in range(20):
        kfold = KFold(n_splits=5, shuffle=True, random_state=seed + i)

        for (tri, tsi), (tri_o, tsi_o) in zip(kfold.split(X), kfold.split(X_o)):
            m = bfgs_ldl()
            m.fit(X[tri], y[tri])
            scores = score(y[tsi], m.predict(X[tsi]))
            df1.loc[len(df1.index)] = scores

            m = bfgs_ldl()
            m.fit(X_o[tri_o], y_o[tri_o])
            # we NEVER modified testing data.
            scores = score(y[tsi], m.predict(X[tsi]))
            df2.loc[len(df2.index)] = scores

    df1.to_excel('bfgs.xlsx', index=False)
    df2.to_excel('ssg-bfgs.xlsx', index=False)
    