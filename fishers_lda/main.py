import pandas as pd
import numpy as np
import visualize
from fisher_lda import LDA

if __name__ == "__main__":

        #####################################      FISHER'S LDA     #########################################

        data = pd.read_csv("data/a1_d1.csv", header = None)
        X = data.iloc[:,:-1]
        T = data.iloc[:,-1]
        X = np.asarray(X)
        X = X.reshape ((X.shape[0], -1))
        T = np.asarray(T)
        X = X.T # columns are individual training examples, rows are the features
        T = T.T # rank one column vector

        model = LDA()
        model.fit(X, T)
        model_prediction = model.predict (X)
        test_accuracy, f_score = model.evaluate (model_prediction, T)
        print("\nACCURACY : {}\nF1-SCORE : {}".format(test_accuracy, f_score))
        
        w = model.get_w()
        discriminant_point = model.get_discriminant_point()
        visualize.visualize (X, T, w, discriminant_point, '1')