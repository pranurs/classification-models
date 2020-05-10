import pandas as pd
import numpy as np
import visualize
from fisher_lda import LDA

def train_test_split (data, train_test = (0.8,0.2)):
        '''
        Performs a train-test-split on the data, in the desired ratio
        '''

        np.random.seed (42)
        data = data.sample(frac = 1)
        X = data.iloc[:,:-1]
        T = data.iloc[:,-1]
        X = np.asarray(X)
        X = X.reshape ((X.shape[0], -1))
        T = np.asarray(T)

        p_train = train_test[0]
        p_test = train_test[1]

        X_train = X[0 : (int)(p_train * X.shape[0]), :]
        T_train = T[0 : (int)(p_train * X.shape[0])]

        X_test = X[(int)(p_train * X.shape[0]) : , :]
        T_test = T[(int)(p_train * X.shape[0]) : ]

        X_train = X_train.T
        X_test = X_test.T
        T_train = T_train.T
        T_test = T_test.T

        return X_train, T_train, X_test, T_test


if __name__ == "__main__":

        #####################################      FISHER'S LDA     #########################################

        data = pd.read_csv("data/a1_d1.csv", header = None)
        
        # Columns of X_train and X_test are individual data examples, rows are feature vectors
        # T_train and T_test are rank one column vectors containing the true class labels
        X_train, T_train, X_test, T_test = train_test_split (data)

        model = LDA()
        model.fit(X_train, T_train)
        model_prediction_test = model.predict (X_test)
        model_prediction_train = model.predict (X_train)
        
        train_accuracy, train_f_score = model.evaluate (model_prediction_train, T_train)
        test_accuracy, test_f_score = model.evaluate (model_prediction_test, T_test)
        
        print("\nTRAIN ACCURACY : {}\nTRAIN F1-SCORE : {}".format(train_accuracy, train_f_score))
        print("\nTEST ACCURACY : {}\nTEST F1-SCORE : {}".format(test_accuracy, test_f_score))
        
        w = model.get_w()
        discriminant_point = model.get_discriminant_point()
        visualize.visualize (X_train, T_train, w, discriminant_point, 'train_1')
        visualize.visualize (X_test, T_test, w, discriminant_point, 'test_1')
