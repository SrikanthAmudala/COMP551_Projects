import numpy as np
import pandas as pd
from submission import config


data = pd.read_csv(config.input_file_path)
# data = pd.read_csv(r"C:\Users\k_mathin\PycharmProjects\Masters\MiniProject2\house-votes-84.csv")

train_size = int(0.75*data.shape[0])
test_size = int(0.25*data.shape[0])
XY_train = data.iloc[0:train_size,:]
N = XY_train.shape[0]
X = data.iloc[:,:-1]
y = data.iloc[:,-1]
#training set split
X_train = X.iloc[0:train_size,:]
y_train = y.iloc[0:train_size]
#testing set split
X_test = X.iloc[train_size:,:]
y_test = y.iloc[train_size:]

#Training
#dimension represented by j for each data point x
#for each class k
pk = []
theta = []
theta_m = []
for item in XY_train['target'].unique():
    data_k = np.asarray(XY_train.loc[XY_train['target']==item])[:,:-1]
    nk = data_k.shape[0]
    data_sum = data_k.sum(axis = 0)
    theta_k = data_sum/nk
    theta_k = np.where(theta_k == 0, 1, theta_k)
    theta_k_m = 1 - theta_k
    theta_k_m = np.where(theta_k_m == 0, 1, theta_k_m)
    theta.append(theta_k)
    theta_m.append(theta_k_m)
    pk.append(nk/N)
#Predict
predict = []
for i in range(0,X_test.shape[0]):
    p_k = []
    for k in range(0,XY_train['target'].unique().shape[0]):
        Px_i = (theta[k]**(X_test.iloc[i].values))*(theta_m[k]**(1 - X_test.iloc[i].values))
        p = Px_i.prod()
        p_k.append(pk[k]*p)
    predict.append(p_k.index(max(p_k)))

from sklearn import metrics

print(metrics.accuracy_score(y_test, predict))