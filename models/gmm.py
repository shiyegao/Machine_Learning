import numpy as np



class GMM:

    def __init__(self, tol=0.0001):
        self.tol = tol

    def phi(self, j, X): # phi: (n_sample, 1)
        self.sigma[j] = self.no_zero(self.sigma[j])
        ex = - 0.5 * np.square(X - self.mu[j]) @ (1 / self.sigma[j]).T
        return np.exp(ex) 

    def no_zero(self, vec, bound=0.0001):
        shape = (vec.shape[0], 1)  if vec.shape[0]>1 else (1, vec.shape[1])
        return np.fmax(vec, np.ones(shape)*bound) 

    def e_step(self, X):
        norm = self.no_zero(np.prod(self.sigma, axis=1))  # norm: (n_class, 1)
        denom = np.sum([self.alpha[j, 0] / norm[j, 0] * self.phi(j, X) \
                for j in range(self.n_class)], axis=0) # denorm: (n_sample, 1)
        denom = self.no_zero(denom)  # norm: (n_class, 1)

        for j in range(self.n_class):
            num = self.alpha[j, 0]  / norm[j, 0] * self.phi(j, X)
            self.gamma[:, j] =  num / denom 
        

    def m_step(self, X):
        gamma = self.no_zero(np.sum(self.gamma.T, axis=1))  # gamma: (n_class, 1)
        
        self.alpha = gamma / self.n_sample    # alpha: (n_class, 1)
        for j in range(self.n_class): 
            self.mu[j] = self.gamma.T[j] @ X / gamma[j, 0]    # mu: (n_class, n_dim)
            self.sigma[j] = self.gamma.T[j] @ np.square(X - self.mu[j]) / gamma[j, 0] # gamma_y: (1, n_dim)


    def random_vec(self, n, sum=1):
        rand_vec = np.mat(np.random.random((1, n)))
        rand_vec = rand_vec / np.sum(rand_vec) * sum
        return rand_vec


    def random_mat(self, shape, sum=1):
        rand_mat = np.mat(np.zeros((1, shape[1])))
        for _ in range(shape[0]):
            rand_mat = np.concatenate((rand_mat, self.random_vec(shape[1], sum=sum)), axis=0)
        return rand_mat[1:, :]


    def fit(self, X, y, epoch=1000):
        n_sample, n_dim = X.shape
        n_class = np.unique(y).shape[0]
        self.n_class = n_class
        self.n_sample = n_sample

        
        # Initialization
        self.alpha = np.mat(self.random_vec(n_class)).T    # alpha: (n_class, 1)
        self.gamma = self.random_mat((n_sample, n_class))   # gamma: (n_sample, n_class)
        self.mu = self.random_mat((n_class, n_dim))          # mu: (n_class, n_dim)
        self.sigma = self.random_mat((n_class, n_dim))      # sigma: (n_class, n_dim)

        # Fitting
        for i in range(1, epoch+1): 
            old_mu = self.mu.copy()
            old_alpha = self.alpha.copy()

            self.e_step(X)    
            self.m_step(X)  

            err = np.sum(np.abs(old_mu - self.mu))      # err: (n_class, n_dim)
            err_alpha = np.sum(np.abs(old_alpha - self.alpha))  # err_alpha: (n_class, 1)
            print("Epoch: [{}/{}], err_mean:{:.7f}, err_alpha:{:.7f}".format(i, epoch, err, err_alpha))
            if (err<=self.tol) and (err_alpha<self.tol):  break
        self.y_dict(X, y)
        return self.mu, self.sigma

    def y_dict(self, X, y):
        vote_mu_to_real = np.mat(np.zeros((self.n_class, self.n_class)))
        for i in range(X.shape[0]):
            vote_mu_to_real[np.argmax(self.gamma[i]), y[i]] += 1
        # print(vote_mu_to_real, np.sum(vote_mu_to_real))
        ydict = np.argmax(vote_mu_to_real, axis=1)
        self.ydict = ydict
        return ydict

    def score(self, X, y):
        y = np.mat(y)
        return np.sum(self.predict(X)==y)/y.shape[0]


    def predict(self, X):
        # print(self.ydict)
        return self.ydict[np.argmin(np.linalg.norm(self.mu, ord=2, axis=1) - X @ self.mu.T, axis=1), 0]



class GMM1:

    def __init__(self, tol=0.01):
        self.tol = tol

    def e_step(self, X, sigma, n_sample, n_class):
        for i in range(n_sample):
            denom = 0
            exp = lambda i,j: np.exp(- 0.5 * (X[i,:] - self.mu[j,:]) * sigma[j].I * \
                    (X[i,:] - self.mu[j,:]).T)
            for j in range(n_class):
                denom += self.alpha[j] * exp(i, j) / np.sqrt(np.linalg.det(sigma[j]))      
            for j in range(n_class):
                num = exp(i, j) / np.sqrt(np.linalg.det(sigma[j]))       
                self.gamma[i,j] = self.alpha[j] * num / denom     

    def m_step(self, X, n_sample, n_class):
        for j in range(n_class):
            gamma, gamma_y, gamma_y_mu = 0, 0, 0   
            for i in range(n_sample):
                gamma_y += self.gamma[i,j] * X[i, :]
                gamma_y_mu += self.gamma[i,j] * (X[i,:] - self.mu[j,:]).T * (X[i,:] - self.mu[j,:])
                gamma += self.gamma[i,j]
            self.mu[j,:] = gamma_y / gamma    
            self.alpha[j] = gamma / n_sample    
            self.sigma[j] = gamma_y_mu / gamma   

    def fit(self, X, y, epoch=1000):

        n_sample, n_dim = X.shape
        n_class = np.unique(y).shape[0]

        # Initialization
        self.alpha = [1 / n_class for _ in range(n_class)]    # alpha: (n_class)
        self.gamma = np.random.random((n_sample, n_class))    # gamma: (n_sample, n_class)
        self.mu = np.mat(np.random.random((n_class, n_dim)))          # mu: (n_class, n_dim)
        self.sigma = [np.mat(np.identity(n_dim)) for _ in range(n_class)] # sigma: (n_class, (n_dim, n_dim))

        # Fitting
        for i in range(1, epoch+1):
            err, err_alpha = 0, 0    
            old_mu = self.mu.copy()
            old_alpha = self.alpha.copy()

            self.e_step(X, self.sigma, n_sample, n_class)    
            self.m_step(X, n_sample, n_class)    

            for z in range(n_class):
                err += (abs(old_mu[z,0] - self.mu[z,0]) + abs(old_mu[z,1] - self.mu[z,1]))     
                err_alpha += abs(old_alpha[z] - self.alpha[z])

            print("Epoch: [{}/{}], err_mean:{:.7f}, err_alpha:{:.7f}".format(i, epoch, err, err_alpha))
            if (err<=self.tol) and (err_alpha<self.tol):  break

        return self.mu, self.sigma



    def score(self, X, y):
        y = np.mat(y)
        return np.sum(self.predict(X)==y)/y.shape[0]


    def predict(self, X):
        return np.argmin(np.linalg.norm(self.mu, ord=2, axis=1) - X @ self.mu.T, axis=1)
