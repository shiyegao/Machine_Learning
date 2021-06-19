# Machine_Learning
Using the SVHN database.  We have logistic regression(+Lasso, +Ridge), SVM(+kernel), GMM, LDA, GAN, CNN, and so on.



# Installation

```bash

git clone https://github.com/shiyegao/Machine_Learning.git
cd Machine_Learning
pip install -r requirements.txt
```


# Usage

For example, you can run logistic regression (lr) with lasso loss using full dataset as follows.
```bash
python main.py --method lr --mode lasso --data vec --dataset full
```


Besides, you can run GAN using full dataset as follows.
```bash
python main.py --method gan --data img --dataset full
```


## There are some parameters which decides the dataset to choose from.

+ --data, choices=['vec', 'img'], whether to use HOG features, or RGB features

+ --dataset, choices=['full', 'part'], whether to use full dataset, or the part


## There are some parameters which decides the model to choose from.
+ --method, choices=['svm','lr', 'lda', 'gmm', 'cnn', 'gan']
+ --mode, choices=['normal','lasso', 'ridge']
+ --kernel, choices=['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']

## There are some parameters which decides the hyper-parameters to choose from.
+ --lr, learning rate
+ --Lambda, used for Lasso and Ridge"
+ --epoch, epoch for training"
+ --batch-size, choose -1 for the whole dataset"

More details can be seen in (utils.py)[utils.py].
