"""
@auth: Sunny
@desc: LDA Working
"""
import pandas
import utils
import collections
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split

df = pandas.read_csv("/Users/Srikanth/PycharmProjects/MiniProject1/breastcancer/clean_breastcancer.csv")


df = df.drop(['Sample code number'], axis=1)

# X = df.iloc[:, 0:-2]
# y = df.iloc[:, -1]
#
# X = np.asarray(X)
# y = np.asarray(y)
#
# # adding one
# X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

X_train, y_train, X_val, y_val = utils.preprocessing(df)


target_count = collections.Counter(y_train)

# class zero

N0 = target_count.get(0.0)
N1 = target_count.get(1.0)

# p(y=0), p(y=1)
p0 = N0 / y_train.shape[0]
p1 = N1 / y_train.shape[0]

mean = [0, 0]

I = np.zeros([y_train.shape[0], 2])
ctr = 0

# mean
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

w0 = np.log(p1 / p0) - 1 / 2 * np.dot(np.dot(mean[0].reshape(-1, 1).T, np.linalg.pinv(covar)), mean[0].reshape(-1, 1))
w = np.dot(np.linalg.pinv(covar), mean[1] - mean[0])

y_pred = ((w0 + np.dot(X_val,w))>0).astype('int64')

accuracy = metrics.accuracy_score(y_val.reshape(-1,1), y_pred.reshape(-1,1))
print(accuracy)