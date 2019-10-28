# Models for Miniproject 1
import numpy as np
import pandas as pd
import utils
import collections
from matplotlib import pyplot as plt


class model(object):
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        return 0

    @staticmethod
    def evaluate_acc(y_pred, y_true):
        return sum(y_pred == y_true) / y_true.size  # in [0,1]

    def do_metrics(self, X_train, y_train, X_val, y_val):
        y_pred_train = self.predict(X_train)
        y_pred_val = self.predict(X_val)
        return [self.evaluate_acc(y_pred=y_pred_train,
                                  y_true=y_train),
                self.evaluate_acc(y_pred=y_pred_val,
                                  y_true=y_val)]


class logistic_regression(model):
    # alpha methods
    @staticmethod
    def _update_alpha_constant(alpha, k):
        return alpha

    def _update_alpha_hyperbolic(self, alpha, k):
        return alpha * (k + 1) / (self.decay * (k + 2))

    # different stopping criterions :
    # epoch : fixed number of iterations
    # convergence : detect convergence, if not default epoch max number
    def _stopping_condition_epoch(self, epoch, **kwargs):
        return epoch != self.num_epoch

    def _stopping_condition_convergence(self, epoch, delta, **kwargs):
        return (np.abs(delta >= self.threshold)) and (epoch != self.num_epoch)

    def __init__(self, m,
                 alpha_mode='hyperbolic', alpha_init=1, decay=1,
                 stopping_mode='convergence', num_epoch=20, threshold=1,
                 train_metrics=False
                 ):
        # m : number of features (dimensions) of the linear model
        self.w = np.random.randn(m)
        self.alpha_init = alpha_init
        self.decay = decay
        self.num_epoch = int(num_epoch)
        self.threshold = threshold
        self.train_metrics = train_metrics

        self.update_alpha = getattr(self, '_update_alpha_' + alpha_mode)
        self.stopping_condition = getattr(self, '_stopping_condition_' + stopping_mode)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # initial step size of the gradient descent
        alpha = self.alpha_init

        # loop control variables
        epoch = 0
        delta = self.threshold

        # record of the training and validation accuracy per epoch
        if self.train_metrics:
            metrics = np.zeros((self.num_epoch + 1, 2))
            metrics[epoch, :] = self.do_metrics(X_train, y_train, X_val, y_val)
            print("epoch {} : train acc. {} val acc. {}\n".format(epoch, metrics[epoch, 0], metrics[epoch, 1]))

        while self.stopping_condition(epoch=epoch, delta=np.linalg.norm(delta)):
            epoch += 1
            dEdw = 0
            for isample in range(y_train.size):
                x_i = X_train[isample]
                y_i = y_train[isample]
                a = np.dot(self.w, x_i)
                dEdw += x_i * (y_i - sigma(a))

            # gradient descent step
            delta = alpha * dEdw
            self.w += delta

            # update step
            alpha = self.update_alpha(alpha, epoch)

            # metrics
            if self.train_metrics:
                metrics[epoch, :] = self.do_metrics(X_train, y_train, X_val, y_val)
                print("epoch {} : train acc. {} val acc. {}\n".format(epoch, metrics[epoch, 0], metrics[epoch, 1]))

        # metrics
        if self.train_metrics:
            if epoch != self.num_epoch:
                metrics = metrics[:epoch, :]

            return metrics

    def predict(self, X):
        return (sigma(np.dot(X, self.w)) > 0.5).astype('int64')


class lda(model):
    def __init__(self, m):
        self.train_metrics = False
        self.w = np.random.randn(m)

    def fit(self, X_train, y_train):
        if X_train.shape[1] == self.w.size:
            X_train = X_train[:, 1:]  # remove bias term !

        target_count = collections.Counter(y_train)
        N0 = target_count.get(0.0)
        N1 = target_count.get(1.0)

        # p(y=0), p(y=1)
        p0 = N0 / y_train.shape[0]
        p1 = N1 / y_train.shape[0]

        I = np.zeros([y_train.shape[0], 2])
        ctr = 0

        # mean
        mean = [0, 0]
        for i, j in zip(y_train, X_train):
            if int(i) == 0:
                mean[0] += j
                I[ctr, 0] = 1
            else:
                mean[-1] += j
                I[ctr, 1] = 1
            ctr += 1

        mean = np.asarray(mean) / [[N0], [N1]]

        # covar
        x_mu_0 = X_train - mean[0]
        x_mu_1 = X_train - mean[1]

        cluster_0 = []
        cluster_1 = []

        for i in range(0, len(I)):
            cluster_0.append(I[i][0] * np.dot(x_mu_0[i].reshape(-1, 1), x_mu_0[i].reshape(-1, 1).T))
            cluster_1.append(I[i][1] * np.dot(x_mu_1[i].reshape(-1, 1), x_mu_1[i].reshape(-1, 1).T))

        cluster_0 = np.asarray(sum(cluster_0))
        cluster_1 = np.asarray(sum(cluster_1))

        covar = (cluster_0 + cluster_1) / (N0 + N1 - 2)

        self.w[0] = np.log(p1 / p0) - 1 / 2 * np.dot(
            np.dot(mean[0].reshape(-1, 1).T, np.linalg.pinv(covar)),
            mean[0].reshape(-1, 1))

        self.w[1:] = np.dot(np.linalg.pinv(covar), mean[1] - mean[0])

    def predict(self, X):
        return ((np.dot(X, self.w)) > 0).astype('int64')


## Package
def kfold(model_init, df, k=5, **params):
    # init
    dataset = df.to_numpy()
    np.random.shuffle(dataset)
    mymodel = model_init(dataset.shape[1], **params)  # remove 1 for target but add 1 for bias

    # metrics
    if mymodel.train_metrics:
        metrics = np.zeros((mymodel.num_epoch + 1, 2))
        # compute baseline for plot
        counts = df.iloc[:, -1].value_counts().sort_values(ascending=False)
        baseline = counts.iloc[0] / (counts.iloc[0] + counts.iloc[1])
    else:
        metrics = np.zeros((1, 2))

    # kfold loop
    folds = np.array_split(dataset, k, axis=0)
    for i in range(k):
        dataset_val = folds[i]
        dataset_train = np.vstack(folds[:i] + folds[i + 1:])
        (X_train, y_train, X_val, y_val) = utils.preprocessing_kfold(dataset_train, dataset_val)

        # metrics
        if mymodel.train_metrics:
            metrics += mymodel.fit(X_train, y_train, X_val, y_val)
        else:
            mymodel.fit(X_train, y_train)
            metrics += mymodel.do_metrics(X_train, y_train, X_val, y_val)

    metrics /= k

    # plot metrics
    if mymodel.train_metrics:
        plt.figure()
        plt.plot(metrics, '.-')
        plt.plot(np.array([0, metrics.shape[0] - 1]), baseline * np.ones(2), '-r')
        plt.xlabel('epochs')
        plt.ylabel('metrics')
        plt.legend(['train acc.', 'val. acc.', 'baseline'])
        plt.grid()
        plt.show()

    return metrics[-1, :]


# sigmoid function
def sigma(x):
    return 1 / (1 + np.exp(-x))
