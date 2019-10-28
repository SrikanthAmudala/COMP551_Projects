# test script

import numpy as np
import pandas as pd
import utils
import model
from matplotlib import pyplot as plt

# params

num_epoch = 400
alpha_init = 1e-2
threshold = 1e-3
decay = 1.2
alpha_mode = 'hyperbolic'
train_metrics = True
stopping_mode = 'convergence'

clean_redwine = pd.read_csv('winequality/clean_redwine.csv',index_col=0)

counts = clean_redwine['quality'].value_counts().sort_values(ascending = False)
baseline = counts.iloc[0]/(counts.iloc[0] + counts.iloc[1])

(X_train,y_train,X_val,y_val) = utils.preprocessing(clean_redwine,prop=0.8)

mymodel = model.logistic_regression(X_train.shape[1], 
                                    num_epoch = num_epoch, 
                                    alpha_init = alpha_init, 
                                    threshold = threshold, 
                                    decay = decay, 
                                    alpha_mode=alpha_mode,
                                    train_metrics=train_metrics,
                                    stopping_mode=stopping_mode)

metrics = mymodel.fit(X_train,y_train,X_val,y_val)

# plot metrics
if train_metrics:
    plt.figure()
    plt.plot(metrics,'.-')
    plt.plot(np.array([0,metrics.shape[0]]), baseline * np.ones(2),'-r')
    plt.xlabel('epochs')
    plt.ylabel('metrics')
    plt.legend(['train acc.','val. acc.','baseline'])
    plt.show()

clean_breastcancer = pd.read_csv('breastcancer/clean_breastcancer.csv',index_col=0)

counts = clean_breastcancer['Class'].value_counts().sort_values(ascending = False)
baseline = counts.iloc[0]/(counts.iloc[0] + counts.iloc[1])

(X_train,y_train,X_val,y_val) = utils.preprocessing(clean_breastcancer,prop=0.8)

mymodel2 = model.logistic_regression(X_train.shape[1], 
                                    num_epoch = num_epoch, 
                                    alpha_init = alpha_init, 
                                    threshold = threshold, 
                                    decay = decay, 
                                    alpha_mode=alpha_mode,
                                    train_metrics=train_metrics,
                                    stopping_mode=stopping_mode)

metrics = mymodel2.fit(X_train,y_train,X_val,y_val)

# plot metrics
if train_metrics:
    plt.figure()
    plt.plot(metrics,'.-')
    plt.plot(np.array([0,metrics.shape[0]]), baseline * np.ones(2),'-r')
    plt.xlabel('epochs')
    plt.ylabel('metrics')
    plt.legend(['train acc.','val. acc.','baseline'])
    plt.show()
