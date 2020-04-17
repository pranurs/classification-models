import numpy as np
import utils
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm, tqdm_notebook

class Layer:

        def __init__ (self, n_nodes, activation, use_bias = True):

                self._n_nodes = n_nodes
                self._activation = activation
                self._use_bias = use_bias

class NeuralNetwork:

        def __init__ (self, learning_rate = 0.0005, n_layers = None, n_layer_nodes = None, layer_activations = None, weights = None, bias_weights = None):

                # Model parameters 
                self._alpha = learning_rate

                if n_layers is not None:
                        self._n_layers = n_layers
                else:
                        self._n_layers = 0

                if n_layer_nodes is not None:
                        self._n_layer_nodes = n_layer_nodes
                else:
                        self._n_layer_nodes = []

                if layer_activations is not None:
                        self._layer_activations = layer_activations
                else:
                        self._layer_activations = []

                if weights is not None:
                        self._weights = weights
                else:
                        self._weights = []

                if bias_weights is not None:
                        self._bias_weights = bias_weights
                else:
                        self._bias_weights = []

                # For internal computation
                self._use_bias = True

                # For performance measurement
                self.test_accuracy = 0
                self.train_accuracy = 0
                self.errors = []

        def add (self, layer):

                if self._n_layers is 0:
                        self._layer_activations.append (layer._activation)
                        self._n_layer_nodes.append(layer._n_nodes)
                        self._n_layers += 1
                        self._use_bias = layer._use_bias
                        return

                self._layer_activations.append (layer._activation)
                self._weights.append (np.random.randn(layer._n_nodes, self._n_layer_nodes[-1]))
                self._n_layer_nodes.append(layer._n_nodes)

                if self._use_bias:
                        self._bias_weights.append(np.random.randn(layer._n_nodes, 1))
                else:
                        self._bias_weights.append(None)

                self._n_layers += 1

        def _forward_propogation (self, input_vector, labels):

                layers = []

                layer = np.reshape(input_vector, (input_vector.shape[0], -1))
                layers.append(layer)

                for i in range (self._n_layers):

                        if i is 0:
                                continue

                        layer = np.matmul (self._weights[i - 1], utils.activate(self._layer_activations[i - 1], layer))

                        if self._bias_weights[i - 1] is not None:
                                layer += self._bias_weights[i - 1]

                        layers.append(layer)

                layers[-1][layers[-1] > 15] = 15
                layers[-1][layers[-1] < -15] = -15

                prediction = utils.activate(self._layer_activations[self._n_layers - 1], layer)
                error = np.sum (-(labels * np.log(prediction) +  (1 - labels) * np.log(1 - prediction)), axis = 1)
                self.errors.append(error)
                # print(error)

                self.train_accuracy = np.sum(((prediction>=0.5) - labels)**2, axis = 1)
                self.train_accuracy = 1 - self.train_accuracy / prediction.shape[1]

                return layers, prediction

        def _back_propogation (self, layers, prediction, labels):

                delta = prediction - labels

                for i in reversed(range(self._n_layers)):

                        if i is 0:
                                break

                        dW = np.matmul(delta, utils.activate(self._layer_activations[i - 1], layers[i - 1]).T)

                        dB = np.sum (delta, axis = 1)
                        dB = np.reshape (dB, (dB.shape[0], -1))

                        delta = utils.calc_der(self._layer_activations[i - 1], layers[i - 1]) * (np.matmul(self._weights[i - 1].T, delta))

                        self._weights[i - 1] -= self._alpha * dW

                        if self._bias_weights[i - 1] is not None:
                                self._bias_weights[i - 1] -= self._alpha * dB
                
        def _adjust_weights (self, input_vector, labels):

                layers, prediction = self._forward_propogation (input_vector, labels)
                self._back_propogation (layers, prediction, labels)

        def fit (self, features, labels, epochs = 40, method = 'BGD'):

                if method == 'BGD':

                        for epoch in tqdm(range(epochs)):

                                self._adjust_weights (features, labels)
                                # print("\nEPOCH {}\n".format(epoch))
                                # print(self.train_accuracy)

                else:

                        for epoch in tqdm(range(epochs)):

                                epoch_accuracy = 0

                                for i in range(features.shape[1]):
                                        self._adjust_weights (features[:,i], labels[:,i])
                                        epoch_accuracy += self.train_accuracy

                                epoch_accuracy = epoch_accuracy/features.shape[1]
                                # print("\nEPOCH {}\n".format(epoch))
                                # print(epoch_accuracy)
                                self.train_accuracy = epoch_accuracy


        def predict (self, input_vector, labels):

                layer = np.reshape(input_vector, (input_vector.shape[0], -1))

                for i in range (self._n_layers):

                        if i is 0:
                                continue

                        layer = np.matmul (self._weights[i - 1], utils.activate(self._layer_activations[i - 1], layer))

                        if self._bias_weights[i - 1] is not None:
                                layer += self._bias_weights[i - 1]

                layer[layer > 15] = 15
                layer[layer < -15] = -15

                prediction = utils.activate(self._layer_activations[self._n_layers - 1], layer)

                self.test_accuracy = np.sum(((prediction>=0.5) - labels)**2, axis = 1)
                self.test_accuracy = 1 - self.test_accuracy / prediction.shape[1]

                return prediction >= 0.5


if __name__ == "__main__":

        np.random.seed (42)

        data = pd.read_csv('housepricedata_NN.csv', header = None)
        X = data.iloc[1:,:-1].astype(int)
        Y = data.iloc[1:,-1].astype(int)
        X = np.array(X)
        Y = np.array(Y)
        Y = np.reshape(Y, (Y.shape[0],1))

        X_min = np.min(X, axis = 0)
        X_max = np.max(X, axis = 0)
        X = (X - X_min)/(X_max - X_min)

        X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size = 0.2, random_state = 42)
        X_train = X_train.T
        X_test = X_test.T
        Y_train = Y_train.T
        Y_test = Y_test.T

        model = NeuralNetwork()
        model.add (Layer (X_train.shape[0], 'relu'))
        model.add (Layer (20, 'relu'))
        model.add (Layer (12, 'relu'))
        model.add (Layer (1, 'sigmoid'))
        
        model.fit(X_train, Y_train, 40, 'SGD')
        model.predict(X_test, Y_test)
        print(model.train_accuracy)
        print (model.test_accuracy)
