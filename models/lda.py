import numpy as np


class LDA:
    def __init__(self, mode='linear'):
        self.mode = mode


    def init_params(self, X, y):
        self.n_sample, self.n_dim = X.shape
        self.cnt_class = np.mat(np.bincount(y.flatten())).T  # class_cnt: (n_class, 1)
        self.n_class = self.cnt_class.shape[0]
        self.priorP = np.mat(self.cnt_class / self.n_sample)  # priorP: (n_class, 1), e.g., [[0.09], [0.11],...,[0.32], [0.08]]

        # covaraint matrix: Sw = n1 * Sigma1 + n2 * Sigma2 + ... + n10 * Sigma10
        sums = np.zeros((self.n_class, self.n_dim))      # means: (n_class, n_dim)
        for i in range(self.n_sample):
            sums[y[i, 0], :] += X[i, :]
        self.mu_i = sums / self.cnt_class           # in-class mean: (n_class, n_dim)
        self.mu = self.priorP.T * self.mu_i    # mu: (1, n_dim)

        # Sw: (n_dim, n_dim)##########
        self.Sw = np.sum([np.cov(X[y.reshape(-1) == c].T) for c in range(self.n_class)], axis=0) / self.n_class 
        
        # Sb: (n_dim, n_dim)##########
        self.Sb = np.sum([(self.mu_i[c]-self.mu).T @ (self.mu_i[c]-self.mu) \
            * self.cnt_class[c, 0] for c in range(self.n_class)], axis=0) / self.n_sample #

        # inverse of Sw
        U, S, V = np.linalg.svd(self.Sw)
        Sn = np.mat(np.diag(S)).I
        self.Sw_inv = V.T @ Sn @ U.T

        return self.Sw, self.Sb

    def fit(self, X, y, n_component=0):

        _, Sb = self.init_params(X, y)
        SS = self.Sw_inv @ Sb

        la, vectors = np.linalg.eig(SS)
        la = np.real(la)
        vectors = np.real(vectors)

        laIdx = np.argsort(-la)

        if n_component == 0:
            n_component = self.n_class - 1

        lambda_index = laIdx[:n_component]
        self.W = vectors[:,lambda_index]
        self.n_component = n_component
        return self.W
        
    def transform(self, X):
        return np.dot(X, self.W)


    def predict_prob(self, X):
        # linear discriminant
        value = X @ self.Sw_inv @ self.mu_i.T + np.log(self.priorP.T)\
                - 0.5 * np.multiply(self.mu_i.T, self.Sw_inv @ self.mu_i.T).sum(axis=0)  # value: (n_sample, n_class)
        return value 


    def predict(self, X):
        return np.argmax(self.predict_prob(X), axis=1)


    def score(self, X, y):
        y = np.mat(y)
        return np.sum(self.predict(X)==y)/y.shape[0]

