import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
from preprocess import preprocess_for_gradient_descent, preprocess_for_naive_bayes
from naive_bayes import NaiveBayes

def k_cross_validation (X, T, K, binary = 0):

        # K-fold cross-validation

        fold_len = len(X)//5
        X_folds = []
        T_folds = []
        k = 0
        results = []
        mean_test_accuracy = 0
        mean_f_score = 0
        stddev_test_accuracy = 0
        stddev_f_score = 0

        for i in range(K):
                X_folds.append(X[k:k+fold_len])
                T_folds.append(T[k:k+fold_len])
                k += fold_len
                if k > len(X):
                        k = len(X)

        for i in range(K):

                X_test = X_folds[i]
                T_test = T_folds[i]
                X_train = []
                T_train = []

                for j in range(K):
                        if j == i:
                                continue
                        X_train = X_train + X_folds[j]
                        T_train = T_train + T_folds[j]

                model = NaiveBayes (alpha = 1)
                prior, likelihood, classes, vocabulary = model.fit (X_train, T_train, binary)
                model_prediction = model.predict (X_test, prior, likelihood, classes, vocabulary)
                test_accuracy, f_score = model.evaluate (model_prediction, T_test)
                results.append((test_accuracy, f_score))
                mean_test_accuracy += test_accuracy
                mean_f_score += f_score
                # print("TRAINING ACCURACY :", model.train_accuracy)
                # print("Run {} : test accuracy = {}, f-score = {}".format(i,test_accuracy,f_score))

        mean_test_accuracy /= K
        mean_f_score /= K

        for i in range(len(results)):

                stddev_test_accuracy += (results[i][0] - mean_test_accuracy)**2
                stddev_f_score += (results[i][1] - mean_f_score)**2
        
        stddev_test_accuracy = np.sqrt(stddev_test_accuracy / len(results))
        stddev_f_score = np.sqrt(stddev_f_score / len(results))

        return results, mean_test_accuracy, mean_f_score, stddev_test_accuracy, stddev_f_score


if __name__ == "__main__":



        #####################################      NAIVE BAYES      #########################################
        
        with open('data/a1_d3.txt') as file:
                data = list(csv.reader(file, delimiter = '\t'))

        reviews, labels = preprocess_for_naive_bayes (data)

        results, mean_test_accuracy, mean_f_score, stddev_test_accuracy, stddev_f_score = k_cross_validation (reviews, labels, K = 5, binary = 1)
        
        # print(results)
        print("TEST ACCURACY : {} +/- {}".format(mean_test_accuracy, stddev_test_accuracy))
        print("F1-SCORE : {} +/- {}".format(mean_f_score, stddev_f_score))

        