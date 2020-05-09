import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
from preprocess import preprocess_for_gradient_descent, preprocess_for_naive_bayes
from neural_network_model import Layer, NeuralNetwork
from logistic_regression import LogisticRegression
from naive_bayes import NaiveBayes
from fisher_lda import LDA

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

def plot_loss (costs, title):

        fig = plt.figure(figsize = (8,8))
        plt.plot(costs)
        plt.xticks(np.arange(0, len(costs), step = (len(costs)//20)), rotation = 45)
        plt.xlabel('Epoch')
        plt.ylabel('Cost')
        plt.title('Loss over epochs')
        plt.savefig('loss_plots/loss_' + title)
        plt.close(fig)


if __name__ == "__main__":


        ########################################      NEURAL NETWORK      ####################################

        data = pd.read_csv('datasets/housepricedata.csv', header = None)
        X_train, X_test, X_val, Y_train, Y_test, Y_val = preprocess_for_gradient_descent (data.iloc[1:,:], 'colwise', 'standardization')
        print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)
        
        # '''
        model = NeuralNetwork(learning_rate=0.0005, initialization = 'gaussian', scaling = 'standard')
        model.add (Layer (X_train.shape[0], 'relu'))
        model.add (Layer (4, 'relu', True, 1))
        model.add (Layer (4, 'relu', True, 0.8))
        model.add (Layer (4, 'relu', True, 1))
        model.add (Layer (4, 'relu', True, 1))
        model.add (Layer (4, 'relu', True, 1))



        # model.add (Layer (4, 'relu', True, 0.8))
        # model.add (Layer (4, 'relu', True, 1))
        # model.add (Layer (4, 'relu', True, 1))
        # model.add (Layer (4, 'relu', True, 1))



        # model.add (Layer (8, 'relu', True, 1))
        # model.add (Layer (8, 'relu', True, 1))
        # model.add (Layer (8, 'relu', True, 0.8))





        # model.add (Layer (12, 'leaky_relu'))
        model.add (Layer (1, 'sigmoid'))
        # '''

        # data = pd.read_csv('datasets/data_banknote_authentication.txt', header = None)
        # X_train, X_test, X_val, Y_train, Y_test, Y_val = preprocess_for_gradient_descent (data, 'colwise', 'min-max')

        # print("\n\nNEURAL NETWORK:\n")
        # model = NeuralNetwork(learning_rate=1.5)
        # model.add (Layer (X_train.shape[0], 'identity'))
        # model.add (Layer (1, 'sigmoid'))

        # '''
        # # train > test:
        # model = NeuralNetwork(0.00005, 'gaussian', 'kaiming')
        # model.add (Layer (X_train.shape[0], 'relu'))
        # model.add (Layer (8, 'relu'))
        # model.add (Layer (8, 'relu'))
        # model.add (Layer (8, 'relu'))
        # model.add (Layer (8, 'relu'))
        # model.add (Layer (1, 'sigmoid'))
        # '''
        

        model.fit (X_train, Y_train, 5000, 'BGD')
        output_val = model.predict(X_val)
        output_test = model.predict (X_test)
        val_accuracy, val_f_score = model.evaluate (output_val, Y_val)
        test_accuracy, test_f_score = model.evaluate (output_test, Y_test)

        print("Train accuracy: ", model.train_accuracy)
        print ("Val accuracy: ", val_accuracy)
        print ("Test accuracy: ", test_accuracy)

        print("\nVal F1-Score: ", val_f_score)
        print("Test F1-Score: ", test_f_score)
        plot_loss (model.costs, 'NN_test')
        # print(model._weights)
        # print(model._bias_weights)
        # # print(model.costs)
        
        ####################################    LOGISTIC REGRESSION   #####################################
        
        # print("\n\nLOGISTIC:\n");

        # data = pd.read_csv('datasets/data_banknote_authentication.txt', header = None)
        # X_train, X_test, X_val, Y_train, Y_test, Y_val = preprocess_for_gradient_descent (data, 'rowwise', 'min-max', (0.7,0.15,0.15))

        # print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)
        # # model = LogisticRegression(2, regularisation='None', lambda_reg=1)
        # model = LogisticRegression (learning_rate=0.005, initialisation='gaussian', regularisation='L1', lambda_reg=0.5)
        # model.fit (X_train, Y_train, 5000, 'BGD')
        # output_val = model.predict(X_val)
        # output_test = model.predict (X_test)
        # val_accuracy, val_f_score = model.evaluate (output_val, Y_val)
        # test_accuracy, test_f_score = model.evaluate (output_test, Y_test)

        # print("Train accuracy: ", model.train_accuracy)
        # print ("Val accuracy: ", val_accuracy)
        # print ("Test accuracy: ", test_accuracy)

        # print("\nVal F1-Score: ", val_f_score)
        # print("Test F1-Score: ", test_f_score)
        # print(model._weights)
        # # print("\n\n", model.costs)
        # plot_loss (model.costs, 'logistic_10000_epochs')

        #####################################      NAIVE BAYES      #########################################
        
        # with open('datasets/a1_d3.txt') as file:
        #         data = list(csv.reader(file, delimiter = '\t'))

        # reviews, labels = preprocess_for_naive_bayes (data)

        # results, mean_test_accuracy, mean_f_score, stddev_test_accuracy, stddev_f_score = k_cross_validation (reviews, labels, K = 5, binary = 1)
        
        # # print(results)
        # print("TEST ACCURACY : {} +/- {}".format(mean_test_accuracy, stddev_test_accuracy))
        # print("F1-SCORE : {} +/- {}".format(mean_f_score, stddev_f_score))

        #####################################      FISHER'S LDA     #########################################

        # data = pd.read_csv("datasets/a1_d2.csv", header = None)
        # X = data.iloc[:,:-1]
        # T = data.iloc[:,-1]
        # X = np.asarray(X)
        # X = X.reshape ((X.shape[0], -1))
        # T = np.asarray(T)
        # X = X.T # columns are individual training examples, rows are the features
        # T = T.T # rank one column vector

        # model = LDA()
        # model.fit(X, T)
        # model_prediction = model.predict (X)
        # test_accuracy, f_score = model.evaluate (model_prediction, T)
        # print("\nACCURACY : {}\nF1-SCORE : {}".format(test_accuracy, f_score))
        # model.visualize (X, T, '2')