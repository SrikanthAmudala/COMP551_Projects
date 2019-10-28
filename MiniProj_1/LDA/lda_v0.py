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

for i, j in zip(y_train, X_train):
    if int(i) == 0:
        mean[0] += j
    else:
        mean[-1] += j

mean = np.asarray(mean)
y1 = X_train - mean[0]
y2 = X_train - mean[-1]


covar = np.dot(y1.T, y2)/(N0+N1-2)

