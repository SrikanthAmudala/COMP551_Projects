import numpy as np
import pandas as pd
import utils
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

path = 'winequality/clean_redwine.csv'
df = pd.read_csv(path,index_col=0)
# df = utils.augment_square(df)
# df = utils.augment_interact(df)

(X_train,y_train,X_val,y_val) = utils.preprocessing(df)

print('logistic')
clf = LogisticRegression(penalty='l2',C = 1e6, solver='lbfgs',multi_class='multinomial').fit(X_train, y_train)
print(clf.score(X_train, y_train))
print(clf.score(X_val, y_val))

print('lda')
clf = LinearDiscriminantAnalysis().fit(X_train, y_train) 
print(clf.score(X_train, y_train))
print(clf.score(X_val, y_val))