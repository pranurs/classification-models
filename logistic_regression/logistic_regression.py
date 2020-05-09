import numpy as np
from tqdm import tqdm, tqdm_notebook
import utils

class LogisticRegression:

        def __init__ (self, learning_rate = 0.005, initialisation = 'gaussian', regularisation = 'None', lambda_reg = 0.01, weights = None):

                if weights is not None:
                        self._weights = weights
                else:
                        self._weights = None

                self._alpha = learning_rate
                self._regularisation = regularisation
                self._lambda_reg = lambda_reg 
                self._initialisation = initialisation
                self.costs = []

                # print("LR = {}, Init = {}, Regularisation = {}, Lambda = {}".format(learning_rate, initialisation, regularisation, lambda_reg))


        def _accuracy (self, model_prediction, labels):

                accuracy = np.sum((model_prediction - labels)**2, axis = 0)
                accuracy = 1 - accuracy / model_prediction.shape[0]
                return accuracy

        def _update_weights (self, X, diff):

                # No. of training examples
                n = X.shape[0]

                if self._regularisation == 'None':

                        # dW = (1/n) * np.matmul(X.T, diff)
                        dW = np.matmul(X.T, diff)


                elif self._regularisation == 'L1':
                        
                        tmp_w = np.copy(self._weights)
                        tmp_w[tmp_w >= 0] = 1
                        tmp_w[tmp_w < 0] = -1
                        # dW = (1/n) * (np.matmul(X.T, diff)) + self._lambda_reg * tmp_w
                        dW = np.matmul(X.T, diff) + self._lambda_reg * tmp_w


                elif self._regularisation == 'L2':

                        # dW = (1/n) * (np.matmul(X.T, diff)) + self._lambda_reg * self._weights
                        dW = np.matmul(X.T, diff) + self._lambda_reg * self._weights


                else:
                        raise Exception ('Invalid regularisation type.')

                self._weights = self._weights - self._alpha * dW

        def _gradient_descent (self, X, T):

                # No. of training examples
                n = X.shape[0]

                # No. of features
                m = X.shape[1]

                if self._weights is None:
                        if self._initialisation == 'gaussian':
                                self._weights = np.random.randn(m, 1)
                        else:
                                self._weights = np.random.uniform(-1,1,(m, 1))

                tmp = np.matmul(X,self._weights)
                tmp[tmp>15] = 15
                tmp[tmp<-15] = -15
                Y = utils.activate ('sigmoid', tmp)
                # cost = (1/n) * np.sum (-(T * np.log(Y) +  (1 - T) * np.log(1 - Y)), axis = 0)
                cost = np.sum (-(T * np.log(Y) +  (1 - T) * np.log(1 - Y)), axis = 0)
                
                if self._regularisation == 'L1':
                        cost += self._lambda_reg * np.sum(abs(self._weights), axis = 0)
                
                if self._regularisation == 'L2':
                        cost += self._lambda_reg * np.sum(self._weights ** 2)

                diff = Y - T

                self.train_accuracy = self._accuracy (Y>=0.5, T)
                self.costs.append(cost)
                # print(cost)

                self._update_weights (X, diff)

        def fit (self, features, labels, epochs = 40, method = 'BGD'):

                features = np.reshape(features, (features.shape[0], -1))
                labels = np.reshape(labels, (labels.shape[0], -1))
                features = np.concatenate ((np.ones((features.shape[0],1)), features), axis = 1)

                if method == 'BGD':

                        for epoch in tqdm(range(epochs)):

                                self._gradient_descent (features, labels)
                                # print("\nEPOCH {}\n".format(epoch))
                                # print(self.train_accuracy)

                else:

                        for epoch in tqdm(range(epochs)):

                                epoch_accuracy = 0

                                for i in range(features.shape[0]):

                                        X = np.reshape(features[i], (features[i].shape[0], -1)).T
                                        T = np.reshape(labels[i], (labels[i].shape[0], -1)).T

                                        self._gradient_descent (X, T)
                                        epoch_accuracy += self.train_accuracy

                                epoch_accuracy = epoch_accuracy/features.shape[0]
                                # print("\nEPOCH {}\n".format(epoch))
                                # print(epoch_accuracy)
                                self.train_accuracy = epoch_accuracy

        def evaluate (self, model_prediction, labels):

                labels = np.reshape(labels, (labels.shape[0], -1))

                test_accuracy = self._accuracy (model_prediction, labels)

                recall = np.sum((model_prediction * labels), axis = 0) / np.sum(labels, axis = 0)
                precision = np.sum((model_prediction * labels), axis = 0) / np.sum(model_prediction, axis = 0)
                f_score = (2*precision*recall) / (precision + recall)

                # print("recall", recall)
                # print("precision", precision)

                return test_accuracy, f_score

        def predict (self, features):

                features = np.reshape(features, (features.shape[0], -1))
                features = np.concatenate ((np.ones((features.shape[0],1)), features), axis = 1)

                tmp = np.matmul(features, self._weights)
                tmp[tmp>15] = 15
                tmp[tmp<-15] = -15
                prediction = utils.activate ('sigmoid', tmp)

                return prediction >= 0.5



