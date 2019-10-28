import numpy as np
import pandas as pd


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def preprocessing(df, prop=0.8):
    # Divide data set and normalize data separately (no information leakage)

    X_train = df.to_numpy()
    np.random.shuffle(X_train)
    isep = round(prop * X_train.shape[0])

    y_val = X_train[isep:, -1]
    y_train = X_train[:isep, -1]

    X_val = X_train[isep:, :-1]
    X_train = X_train[:isep, :-1]

    X_val = (X_val - np.mean(X_val, axis=0)) / np.std(X_val, axis=0)
    X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)

    X_val = np.concatenate([np.ones((X_val.shape[0], 1)), X_val], axis=1)
    X_train = np.concatenate([np.ones((X_train.shape[0], 1)), X_train], axis=1)

    return (X_train, y_train, X_val, y_val)


def preprocessing_kfold(dataset_train, dataset_val):
    # Divide data set and normalize data separately (no information leakage)
    y_val = dataset_val[:, -1]
    y_train = dataset_train[:, -1]

    X_val = dataset_val[:, :-1]
    X_train = dataset_train[:, :-1]

    X_val = (X_val - np.mean(X_val, axis=0)) / np.std(X_val, axis=0)
    X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)

    X_val = np.concatenate([np.ones((X_val.shape[0], 1)), X_val], axis=1)
    X_train = np.concatenate([np.ones((X_train.shape[0], 1)), X_train], axis=1)

    return (X_train, y_train, X_val, y_val)

def augment_square(df):
    dataset = df.copy()
    columns = dataset.columns[:-1]
    index = len(columns)
    for column in columns[::-1]:
        dataset.insert(index,column + '^2',np.square(dataset[column]))
    return dataset

def augment_interact(df):
    dataset = df.copy()
    columns = dataset.columns[-2::-1]
    index = len(columns)
    for i in range(len(columns)):
        for j in range(i+1):
            dataset.insert(index,columns[i] + '*' + columns[j],dataset[[columns[i],columns[j]]].product(axis=1))
    
    return dataset
