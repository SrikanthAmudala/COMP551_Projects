
One of the applications of machine learning is in
the classification where model construction is utilized for text
classification. The more the model learns from previous data, the
more accurate it would perform. The ability of a classification
model to achieve high accuracy in text mining is of significant
importance. The complexity of text data unfavourable for most
models, which is a major challenge, the training and testing data
in the classification of data must be selected in a way that the
model enjoys the most efficient learning from previous data and
the highest accuracy in text classification. In this study, the Reddit
dataset of comments, the models for predicting the categories are
developed based on logistic regression, Support Vector Machine
and Naive Bayesian methods and the accuracy of each model is
evaluated. The best performance and accuracy, while it depends
on the nature and complexity of the dataset.


**config.py** 
Contains the input path for the datasets.

**data_preprocessing.py**
Contains the preprocessing steps to process the csv 

**navie_bayes.py** 
Contains the implementation of Bernoulli Navie Bayes from scratch using a binomial data set.

**text_classification.py**
Contains the steps to train the reddit data with different classifiers. All the different classifiers are commented. We have uncommented multinomial binomial for the final submission as it has better accuracy.
This used data_preprocessing.py to pre-process the data.


**xlnet_reddit.ipynb**
Is the python note book that explains our final model that resulted in the better accuracy than every other algorithm we have used.
The note book took 5 hours on a 1050Ti GPU and a machine with 32 GB ram for training, so we have included the python notebook that showcases the results that we were able to achieve.
Please contact us if there are any questions related to the implementation.



