
import numpy as np
from matplotlib.pyplot import axis
from sklearn import svm, preprocessing

# My tool
from utils import load_parser, process_data

# My models
from models import MyNet
from models import logisticRegression
from models import LDA
from models import GMM
from models import GAN




if __name__ == "__main__":

    args = load_parser()

    # process_data(0.1, args.data)
    if args.data=='vec':
        print('\n' + '-'*20 + "  Data Loading  " + '-'*20)
        data = np.load("data/vec/data_324_01.npz") if args.dataset=='part' else np.load("data/vec/data_324_0.npz") 
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']
        print("Training", X_train.shape, y_train.shape)
        print("Testing", X_test.shape, y_test.shape)

        print('\n' + '-'*20 + "  Data Preprocessing  " + '-'*20)
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        scaler = preprocessing.StandardScaler().fit(X_test)
        X_test = scaler.transform(X_test)
    elif args.data=='img':
        data = np.load("data/img/data_01.npz") if args.dataset=='part' else np.load("data/img/data_0.npz") 
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']
        print("Training", X_train.shape, y_train.shape)
        print("Testing", X_test.shape, y_test.shape)


    print('\n' + '-'*20 + "  Training  " + '-'*20)
    if args.method=='lr': 
        model = logisticRegression(
            lr=args.lr, 
            epoch=args.epoch, 
            log_interval=args.log_interval,
            mode=args.mode,
            Lambda=args.Lambda,
            batch_size=args.batch_size
        )
        model.experiment(X_train, y_train, X_test, y_test)
        # model.fit(X_train, y_train)
        print("[{}]    Test_acc: {}".format(args.method, model.score(X_test, y_test)))
    elif args.method == 'svm':
        model = svm.SVC(kernel=args.kernel, gamma='scale')
        model.fit(X_train, y_train.flatten())
        print("[{}, {}]    Test_acc: {}".format(args.method, args.kernel, 
            model.score(X_test, y_test.flatten())))
    elif args.method == 'lda':
        model  = LDA()
        model.fit(X_train, y_train)
        print("[{}]    Test_acc: {}".format(args.method, model.score(X_test, y_test)))
    elif args.method == 'gmm':
        model = GMM()
        model.fit(X_train, y_train)
        print("[{}]    Test_acc: {}".format(args.method, model.score(X_test, y_test)))
    elif args.method == 'cnn':
        model = MyNet().cuda()
        model.run(data, n_epoch=args.epoch, batch_size=args.batch_size, lr=args.lr)
    elif args.method =='gan':
        model = GAN(args)
        model.run(data)
    else:
        print("Model type non-existing. Try again.")
        exit(-1)
