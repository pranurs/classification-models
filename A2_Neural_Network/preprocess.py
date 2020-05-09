import string
import numpy as np

def preprocess_for_gradient_descent (data, structure = 'colwise', feature_scaling = 'min-max', train_val_test = (0.7, 0.15, 0.15)):

        np.random.seed (100)
        data = data.sample(frac = 1)

        X = data.iloc[:,:-1].astype(int)
        Y = data.iloc[:,-1].astype(int)
        X = np.array(X)
        Y = np.array(Y)
        Y = np.reshape(Y, (Y.shape[0],1))

        p_train = train_val_test[0]
        p_val = train_val_test[1]
        p_test = train_val_test[2]

        X_train = X[0 : (int)(p_train * X.shape[0]), :]
        Y_train = Y[0 : (int)(p_train * X.shape[0]), :]

        X_val = X[(int)(p_train * X.shape[0]) : (int)((p_train + p_val) * X.shape[0]), :]
        Y_val = Y[(int)(p_train * X.shape[0]) : (int)((p_train + p_val) * X.shape[0]), :]

        X_test = X[(int)((p_train + p_val) * X.shape[0]) : X.shape[0], :]
        Y_test = Y[(int)((p_train + p_val) * X.shape[0]) : X.shape[0], :]

        if feature_scaling == 'min-max':
                X_min = np.min(X_train, axis = 0)
                X_max = np.max(X_train, axis = 0)
                X_train = (X_train - X_min)/(X_max - X_min)
                X_val = (X_val - X_min)/(X_max - X_min)
                X_test = (X_test - X_min)/(X_max - X_min)

        elif feature_scaling == 'standardization':
                X_mean = np.sum(X_train, axis = 0)/X_train.shape[0]
                X_stddev = np.sqrt(np.sum((X_train - X_mean)**2, axis = 0)/X_train.shape[0])
                X_train = (X_train - X_mean)/X_stddev
                X_val = (X_val - X_mean)/X_stddev
                X_test = (X_test - X_mean)/X_stddev

        else:
                raise Exception ('Invalid feature scaling argument.')

        if structure == 'colwise':
                X_train = X_train.T
                X_test = X_test.T
                X_val = X_val.T
                Y_train = Y_train.T
                Y_test = Y_test.T
                Y_val = Y_val.T

        return X_train, X_test, X_val, Y_train, Y_test, Y_val

def preprocess_for_naive_bayes (data):

        np.random.seed(42)
        np.random.shuffle (data)
        labels = [int(example[-1]) for example in data]
        reviews = [example[:-1] for example in data]

        # Convert each review to lowercase
        reviews = [' '.join(sentence[0].lower().split()) for sentence in reviews]

        # Remove punctuation
        table = str.maketrans('', '', string.punctuation)
        reviews = [sentence.translate(table) for sentence in reviews]

        # Convert each review from a string into a list of words
        reviews = [sentence.split() for sentence in reviews]

        return reviews, labels

