import collections

import numpy as np
import pandas
import utils
from sklearn import metrics


class Lda:
    def __init__(self):
        self.mean = [0, 0]

    def fit(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        target_count = collections.Counter(y_train)

        # class zero

        self.N0 = target_count.get(0.0)
        self.N1 = target_count.get(1.0)

        # p(y=0), p(y=1)
        self.p0 = self.N0 / self.y_train.shape[0]
        self.p1 = self.N1 / self.y_train.shape[0]

        I = np.zeros([self.y_train.shape[0], 2])
        ctr = 0

        # mean
        for i, j in zip(self.y_train, self.X_train):
            if int(i) == 0:
                self.mean[0] += j
                I[ctr, 0] = 1
            else:
                self.mean[-1] += j
                I[ctr, 1] = 1
            ctr += 1

        self.mean = np.asarray(self.mean) / [[self.N0], [self.N1]]

        # covar
        x_mu_0 = X_train - self.mean[0]
        x_mu_1 = X_train - self.mean[1]

        cluster_0 = []
        cluster_1 = []

        for i in range(0, len(I)):
            cluster_0.append(I[i][0] * np.dot(x_mu_0[i].reshape(-1, 1), x_mu_0[i].reshape(-1, 1).T))
            cluster_1.append(I[i][1] * np.dot(x_mu_1[i].reshape(-1, 1), x_mu_1[i].reshape(-1, 1).T))

        cluster_0 = np.asarray(sum(cluster_0))
        cluster_1 = np.asarray(sum(cluster_1))

        self.covar = (cluster_0 + cluster_1) / (self.N0 + self.N1 - 2)

        self.w0 = np.log(self.p1 / self.p0) - 1 / 2 * np.dot(
            np.dot(self.mean[0].reshape(-1, 1).T, np.linalg.pinv(self.covar)),
            self.mean[0].reshape(-1, 1))
        self.w = np.dot(np.linalg.pinv(self.covar), self.mean[1] - self.mean[0])

    def predict(self, X_val):
        y_pred = ((self.w0 + np.dot(X_val, self.w)) > 0).astype('int64')
        return y_pred


obj = Lda()
df = pandas.read_csv("/Users/Srikanth/PycharmProjects/MiniProject1/winequality/clean_redwine.csv")
# df = df.drop(['Sample code number'], axis=1)


X_train, y_train, X_val, y_val = utils.preprocessing(df)


obj.fit(X_train, y_train, X_val, y_val)
y_predit = obj.predict(X_val)

accuracy = metrics.accuracy_score(y_val.reshape(-1, 1), y_predit.reshape(-1, 1))
print(accuracy)
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]


    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
plot_confusion_matrix(y_val.reshape(-1, 1), y_predit.reshape(-1, 1), classes=np.asarray(['benign', 'malignant']),
                      title='Confusion matrix, without normalization')

plt.show()