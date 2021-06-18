import numpy as np


class logisticRegression:
    
    def __init__(self, lr=0.001 , epoch=100, log_interval=10, mode='lasso', Lambda=0.1, batch_size=-1):
        self.lr = lr  # learning rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.log_interval = log_interval
        self.Lambda = Lambda
        self.mode=mode
        self.acc = []
        self.log = [     1,       2,      5,     10,     20,     50,     100,    150,    200,    250,    300,    350,    400,    450,     500]


    def softmax(self, X):
        return 1.0 / (1 + np.exp(-X/100))


    def sigmoid(self, A, mode='scale', ratio=100):
        if mode=='scale':
            A = A.clip(-10000,100000)
            exp_mat = np.exp(-A/ratio)
        elif mode=='clip':
            exp_mat = np.exp(-A.clip(-100,100000))
        return exp_mat / np.sum(exp_mat, axis=1)


    def get_batch(self, X, i):
        X_batch = X[i*self.batch_size:(i+1)*self.batch_size, :]
        X_batch = np.concatenate((X_batch, np.zeros((self.batch_size-X_batch.shape[0], X_batch.shape[1]))), 0)
        return X_batch


    def gradient(self, X, W, y, mode='normal'):
        if mode=='normal':
            prediction = self.softmax(X @ W)  # h: (batch_size, n_class)
            gradient = X.T * (y - prediction) 
        elif mode=='ridge':
            prediction = self.softmax(X @ W)  # h: (batch_size, n_class)
            gradient = X.T * (y - prediction) + 2 * self.Lambda * W
        elif mode=='lasso':
            prediction = self.softmax(X @ W)  # h: (batch_size, n_class)
            gradient = X.T * (y - prediction) + self.Lambda * np.sign(W)
        return gradient


    def fit(self, X, y):
        '''
            :param X: data, shape = (n_sample, n_dim)
            :param y: label, shape =  (n_sample, 1)
            :return: weights, shape = (n_dim, 1)
        '''
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        n_sample, n_dim = X.shape
        self.batch_size = n_sample
        n_class = len(np.unique(y))
        y_mat = np.zeros((n_sample, n_class))
        for i in range(y.shape[0]):
            y_mat[i, y[i,0]] = 1
        W = np.mat(np.ones((n_dim, n_class)))
        batch = n_sample//self.batch_size + 1
        error = np.zeros((self.batch_size, n_class))

        for e in range(1, self.epoch+1):
            for i in range(batch):
                X_batch = self.get_batch(X, i)        # X_batch: (batch_size, n_dim)
                y_batch = self.get_batch(y_mat, i)    # y_batch: (batch_size, 1)
                gradient_loss = self.gradient(X_batch, W, y_batch, mode=self.mode) # loss: (batch_size, n_class)
                W += -self.lr * gradient_loss       # W: (n_dim, n_class)
            if e % self.log_interval==0:
                h = self.softmax(X @ W)
                acc = np.sum(np.argmax(h, axis=1)==y)/y.shape[0]
                print("Epoch: [{}/{}]    Train_acc:{:.5f}    Mean_loss:{:.2f}  ".format(e, self.epoch, acc, np.sum(error/n_sample)))
        self.W = W
        return W
 

    def experiment(self, X, y, X_, y_):
        '''
            :param X: data, shape = (n_sample, n_dim)
            :param y: label, shape =  (n_sample, 1)
            :return: weights, shape = (n_dim, 1)
        '''
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        n_sample, n_dim = X.shape
        self.batch_size = n_sample
        n_class = len(np.unique(y))
        y_mat = np.zeros((n_sample, n_class))
        for i in range(y.shape[0]):
            y_mat[i, y[i,0]] = 1
        W = np.mat(np.ones((n_dim, n_class)))
        batch = n_sample//self.batch_size + 1
        error = np.zeros((self.batch_size, n_class))

        for e in range(1, self.epoch+1):
            for i in range(batch):
                X_batch = self.get_batch(X, i)        # X_batch: (batch_size, n_dim)
                y_batch = self.get_batch(y_mat, i)    # y_batch: (batch_size, 1)
                gradient_loss = self.gradient(X_batch, W, y_batch, mode=self.mode) # loss: (batch_size, n_class)
                W += -self.lr * gradient_loss       # W: (n_dim, n_class)
            if e % self.log_interval==0:
                h = self.softmax(X @ W)
                train_acc = np.sum(np.argmax(h, axis=1)==y)/y.shape[0]
                self.W = W
                test_acc = self.score(X_, y_)
                print("Epoch: [{}/{}]    Test_acc:{:.5f}    Train_acc:{:.5f}    Mean_loss:{:.2f}  ".format(\
                        e, self.epoch, test_acc, train_acc, np.sum(error/n_sample)))
            # if e in self.log:
            #     self.acc.append(round(test_acc, 4))
        self.W = W
        print(self.acc)
        print(np.max(W, axis=0), np.sum(W), np.sum(W<0.001))
        return W
            
    def predict(self, X):
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        return np.argmax(self.softmax(X @ self.W), axis=1)


    def score(self, X, y):
        y = np.mat(y)
        return np.sum(self.predict(X)==y)/y.shape[0]


