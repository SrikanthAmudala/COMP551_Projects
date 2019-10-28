import data_preprocessing
import pandas
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from submission import config
train_set = pandas.read_csv(config.reddit_train)
test_set = pandas.read_csv(config.reddit_test)

train_set, test_set = data_preprocessing.main(train_set, test_set)

data = train_set['comments']
target = train_set['subreddits']
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.1, random_state=42)

# ALL THE MODELS THAT WERE USED TO COMPARE THE MODEL

# # 54
# text_clf = Pipeline([('vect', CountVectorizer()),
#                      ('tfidf', TfidfTransformer()),
#                      ('clf', LinearSVC()),
#                      ])
#
# # 43
# text_clf_gradientBoosting = Pipeline([('vect', CountVectorizer()),
#                      ('tfidf', TfidfTransformer()),
#                      ('clf', GradientBoostingClassifier(n_estimators=50,verbose=2)),
#                      ])
#
# # 54
# text_clf_lr = Pipeline([('vect', CountVectorizer()),
#                      ('tfidf', TfidfTransformer()),
#                      ('clf', LogisticRegression()),
#                      ])
#
#
# # 7
# text_clf_knn= Pipeline([('vect', CountVectorizer()),
#                      ('tfidf', TfidfTransformer()),
#                      ('clf', KNeighborsClassifier()),
#                      ])
#
# # 47
# text_clf_randomForestdf= Pipeline([('vect', CountVectorizer()),
#                      ('tfidf', TfidfTransformer()),
#                      ('clf', RandomForestClassifier(n_estimators=100)),
#                      ])
#
#
# # 42
# text_clf_NearestCentroid= Pipeline([('vect', CountVectorizer()),
#                      ('tfidf', TfidfTransformer()),
#                      ('clf', NearestCentroid()),
#                      ])

from sklearn.naive_bayes import MultinomialNB

text_clf_multinomialNB = Pipeline([('vect', CountVectorizer()),
                                   ('tfidf', TfidfTransformer()),
                                   ('clf', MultinomialNB()),
                                   ])

text_clf_multinomialNB.fit(X_train, y_train)
predictions_multinomialNB = text_clf_multinomialNB.predict(X_test)
print(metrics.accuracy_score(y_test, predictions_multinomialNB))

# full data train
text_clf_multinomialNB.fit(data, target)
final_test_set = test_set['comments']
predictions = text_clf_multinomialNB.predict(final_test_set)
predictions_df = pandas.DataFrame(predictions)
predictions_df.to_csv("/Users/Srikanth/PycharmProjects/MiniProject1/miniproj2/output/linearSVC_prediction_final.csv",
                      header=['Category'])
