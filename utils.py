
import argparse
import scipy.io  
import numpy as np
from skimage.feature import hog


def get_hog_feature(data):
    return np.array( [hog(
            data[:,:,:,i], cells_per_block=(1,1)) 
            for i in range(data.shape[3]
        )])



def process_data(ratio=1, data='vec'):

    train_data = scipy.io.loadmat('data/train_32x32.mat')  
    X_train, y_train = train_data['X'], train_data['y']-1   # np.array
    X_train, y_train = X_train[:, :, :, :int(ratio*X_train.shape[3])], y_train[:int(ratio*y_train.shape[0])]
    X_train = get_hog_feature(X_train) if data=='vec' else X_train.transpose((3, 0, 1, 2))
    print("Training", X_train.shape, y_train.shape)

    test_data = scipy.io.loadmat('data/test_32x32.mat')  
    X_test, y_test = test_data['X'], test_data['y']-1    # np.array
    X_test, y_test = X_test[:, :, :, :int(ratio*X_test.shape[3])], y_test[:int(ratio*y_test.shape[0])]
    X_test = get_hog_feature(X_test) if data=='vec' else X_test.transpose((3, 0, 1, 2))
    print("Testing", X_test.shape, y_test.shape)

    if data=='vec':
        path = "data/{}/data_{}_{}.npz".format(data, X_train.shape[1],'0'+str(ratio)[2:])
    else:
        path = "data/{}/data_{}.npz".format(data, '0'+str(ratio)[2:])

    np.savez(path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)


def load_parser():

    # Arguments
    parser = argparse.ArgumentParser(description="Machine Learning")

    #################### dataset ####################
    parser.add_argument('--data', type=str, default='vec',
            choices=['vec', 'img'])
    parser.add_argument('--dataset', type=str, default='part',
            choices=['full', 'part'])

    #################### method ####################
    parser.add_argument('--method', type=str, default='lr',
            choices=['svm','lr', 'lda', 'gmm', 'cnn', 'gan'])
    parser.add_argument('--mode', type=str, default='normal',
            choices=['normal','lasso', 'ridge'])
    parser.add_argument('--kernel', type=str, default='rbf', help="Used only when SVM",
            choices=['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'])

    #################### parameters ####################
    parser.add_argument('--lr', type=float, default=0.001,
            help="Learning rate")
    parser.add_argument('--Lambda', type=float, default=10,
            help="Used for Lasso and Ridge")
    parser.add_argument('--epoch', type=int, default=200, 
            help="Epoch for training")
    parser.add_argument('--batch-size', type=int, default=128, 
            help="choose -1 for the whole dataset")
    parser.add_argument('--log-interval', type=int, default=1, 
            help="Interval for logging")
    parser.add_argument("--save_model_interval", type=int, default=50, 
                        help="interval between model saving")

    #################### Gan ####################
    parser.add_argument("--channels", type=int, default=3, 
                        help="number of image channels")
    parser.add_argument("--gan-z-dim", type=int, default=100,
                        help="noise dimension")
    parser.add_argument("--gan-save", type=bool, default=True, 
                        help="whether to save generative images")
    parser.add_argument("--save_interval", type=int, default=10, 
                        help="interval between image saving")

    parser.add_argument("--tag", type=str, default="default",
                        help="tag")
    parser.add_argument("--gan-iters", type=int, default=3000,      
                        help="iters of GAN")



    args = parser.parse_args() 
    return args