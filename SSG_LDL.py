from sklearn.neighbors import NearestNeighbors

import numpy as np
import random

class SSG_LDL:

    def __init__(self, n=300, k=5, fx=0.5, fy=0.5, random_state=None):
        if random_state != None:
            np.random.seed(random_state)
        
        self.n = n
        self.k = k
        self.fx = fx
        self.fy = fy

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

    def fit_transform(self, s):
        self.X, self.y = s['features'], s['labels']
        self.dist = self.fx * np.sum(np.linalg.norm(np.repeat([self.X], self.X.shape[0], axis=0)
            .transpose(1, 0, 2) - self.X, axis=2), axis=1) / self.X.shape[0] + \
            self.fy * np.sum(np.linalg.norm(np.repeat([self.y], self.y.shape[0], axis=0)
            .transpose(1, 0, 2) - self.y, axis=2), axis=1) / self.y.shape[0]

        self.knn = NearestNeighbors(n_neighbors=self.k)
        self.knn.fit(np.concatenate([self.X, self.y], axis=1))

        self.X_ = self.X
        self.y_ = self.y
    
        t = self.X.shape[0] * self.n // 100

        for _ in range(t):
            self.create_synthetic_sample(self.select_sample())

        return {'features': self.X_, 'labels': self.y_}
    