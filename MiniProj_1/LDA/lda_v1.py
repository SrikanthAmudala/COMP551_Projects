# LDA

import pandas
import utils
import collections
import numpy as np

df = pandas.read_csv("/Users/Srikanth/PycharmProjects/MiniProject1/breastcancer/clean_breastcancer.csv")
df = df.drop(['Sample code number'], axis=1)

X_train, y_train, X_val, y_val = utils.preprocessing(df)

target_count = collections.Counter(y_train)

# class zero

N0 = target_count.get(0.0)
N1 = target_count.get(1.0)

p0 = N0 / y_train.shape[0]

# class one
p1 = N1 / y_train.shape[0]

mean = [0, 0]

I = np.zeros([y_train.shape[0],2])
ctr = 0

for i, j in zip(y_train, X_train):
    if int(i) == 0:
        mean[0] += j
        I[ctr,0] = 1
    else:
        mean[-1] += j
        I[ctr,1] = 1
    ctr += 1

mean = np.asarray(mean)/[[N0],[N1]]
x_mu_1 = X_train - mean[0]
x_mu_1 = np.diagonal(np.matmul(x_mu_1,x_mu_1.transpose()))
x_mu_2 = X_train - mean[1]


x_mu_2 = np.diagonal(np.matmul(x_mu_2,x_mu_2.transpose()))
x_mu = np.asarray([x_mu_1,x_mu_2])

sigma = ((I*x_mu.transpose())/(y_train.shape[0]-2)).sum()

w0 = np.log(p1/p0) - np.matmul(mean[1]/sigma,mean[1].transpose()) + np.matmul(mean[0]/sigma,mean[0].transpose()) # 1/2 ?
w0 = np.resize(w0,[y_val.shape[0],1])
w = (mean[1]-mean[0])/sigma
#w = np.resize(w,[y_val.shape[0],w.shape[0]])
w.resize([w.shape[0],1])
y_pred = ((w0 + np.matmul(X_val,w))>0).astype('int64')
y_val.resize([y_val.shape[0],1])
err = abs(y_pred-y_val).sum()
Accuracy = (y_val.shape[0]-err)/y_val.shape[0]

print("Accuracy: "+ str(Accuracy*100)+"%")