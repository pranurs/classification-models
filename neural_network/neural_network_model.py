import numpy as np
import pandas as pd
from tqdm import tqdm, tqdm_notebook
import utils

class Layer:

        def __init__ (self, n_nodes, activation, use_bias = True, keep_prob = 1):

                self._n_nodes = n_nodes
                self._activation = activation
                self._use_bias = use_bias
                self._keep_prob = keep_prob

class NeuralNetwork:

        def __init__ (self, learning_rate = 0.0005, initialization = 'gaussian', scaling = 'standard', regularisation = 'None', lambda_reg = 0.001, n_layers = None, n_layer_nodes = None, layer_activations = None, weights = None, bias_weights = None, keep_probs = None):

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

                if keep_probs is not None:
                        self._keep_probs = keep_probs
                else:
                        self._keep_probs = []

                self._regularisation = regularisation
                self._lambda_reg = lambda_reg 
                self._initialization = initialization
                self._scaling = scaling

                # For internal computation
                self._use_bias = True

                # For performance measurement
                self.train_accuracy = 0
                self.costs = []

                print("LR = {}, Initialization = {}, Scaling = {}, Regularisation = {}, Lambda = {}".format(learning_rate, initialization, scaling, regularisation, lambda_reg))

        def add (self, layer):

                if self._n_layers is 0:
                        self._layer_activations.append (layer._activation)
                        self._n_layer_nodes.append(layer._n_nodes)
                        self._keep_probs.append(layer._keep_prob)
                        self._n_layers += 1
                        self._use_bias = layer._use_bias
                        return

                self._layer_activations.append (layer._activation)

                if self._initialization == 'gaussian':
                        w = np.random.randn(layer._n_nodes, self._n_layer_nodes[-1])

                else: # uniform initialization
                        w = np.random.uniform(-1, 1, (layer._n_nodes, self._n_layer_nodes[-1]))

                if self._scaling == 'standard':
                        w *= np.sqrt(1. / self._n_layer_nodes[-1])

                elif self._scaling == 'kaiming':
                        w *= np.sqrt(2. / self._n_layer_nodes[-1])

                elif self._scaling == 'xavier':
                        w *= np.sqrt(1./ (self._n_layer_nodes[-1] + layer._n_nodes))

                elif self._scaling == 'he': # He et al initialization
                        w *= np.sqrt(2./ (self._n_layer_nodes[-1] + layer._n_nodes))

                else: # Automate according to common best practices

                        if layer._activation == 'relu':
                                w *= np.sqrt(2. / self._n_layer_nodes[-1])

                        elif layer._activation == 'tanh':
                                w *= np.sqrt(1./ (self._n_layer_nodes[-1] + layer._n_nodes))
                                # w *= np.sqrt(1. / self._n_layer_nodes[-1])

                        else:
                                w *= np.sqrt(2./ (self._n_layer_nodes[-1] + layer._n_nodes))

                self._weights.append (w)
                self._keep_probs.append(layer._keep_prob)
                self._n_layer_nodes.append(layer._n_nodes)

                if self._use_bias:
                        self._bias_weights.append(np.random.randn(layer._n_nodes, 1))
                else:
                        self._bias_weights.append(None)

                self._n_layers += 1

        def _forward_propogate (self, input_vector, labels):

                layers = []

                layer = np.reshape(input_vector, (input_vector.shape[0], -1))
                layers.append(layer)

                for i in range (self._n_layers):

                        if i is 0:
                                continue

                        active_layer = utils.activate(self._layer_activations[i - 1], layer)
                        drop = np.random.rand(active_layer.shape[0], active_layer.shape[1]) < self._keep_probs[i-1]
                        active_layer *= drop
                        active_layer /= self._keep_probs[i-1]

                        layer = np.matmul (self._weights[i - 1], active_layer)

                        if self._bias_weights[i - 1] is not None:
                                layer += self._bias_weights[i - 1]
                        
                        layers.append(layer)

                layers[-1][layers[-1] > 15] = 15
                layers[-1][layers[-1] < -15] = -15

                # print(layers)

                prediction = utils.activate(self._layer_activations[self._n_layers - 1], layer)
                cost = np.sum (-(labels * np.log(prediction) +  (1 - labels) * np.log(1 - prediction)), axis = 1)
                self.costs.append(cost)
                # print(cost)

                self.train_accuracy = np.sum(((prediction>=0.5) - labels)**2, axis = 1)
                self.train_accuracy = (1 - self.train_accuracy / prediction.shape[1]) * 100

                return layers, prediction

        def _back_propogate (self, layers, prediction, labels):

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

                layers, prediction = self._forward_propogate (input_vector, labels)
                self._back_propogate (layers, prediction, labels)

        def fit (self, features, labels, epochs = 40, method = 'BGD', mini_batch_size = 128):

                if method == 'BGD':

                        for epoch in tqdm(range(epochs)):

                                self._adjust_weights (features, labels)
                                # print("\nEPOCH {}\n".format(epoch))
                                # print(self.train_accuracy)

                elif method == 'MBGD':

                        n_training_examples = features.shape[1]
                        # print(labels.shape)
                        costs = []

                        for epoch in tqdm(range(epochs)):

                                for i in range(n_training_examples//mini_batch_size):
                                        
                                        self._adjust_weights (features[:, (i*mini_batch_size) : min(n_training_examples, (i+1)*mini_batch_size)], labels[:, (i*mini_batch_size) : min(n_training_examples, (i+1)*mini_batch_size)])

                                costs.append(self.costs[-1]) 
                                # print("\nEPOCH {}\n".format(epoch))
                                # print(self.train_accuracy)
                        self.costs = costs

                else:   
                        #SGD
                        for epoch in tqdm(range(epochs)):

                                epoch_accuracy = 0

                                for i in range(features.shape[1]):
                                        self._adjust_weights (features[:,i], labels[:,i])
                                        epoch_accuracy += self.train_accuracy

                                epoch_accuracy = (epoch_accuracy/features.shape[1]) * 100
                                # print("\nEPOCH {}\n".format(epoch))
                                # print(epoch_accuracy)
                                self.train_accuracy = epoch_accuracy 
                                

        def evaluate (self, model_prediction, labels):

                test_accuracy = np.sum((model_prediction - labels)**2, axis = 1)
                test_accuracy = (1 - test_accuracy / model_prediction.shape[1]) * 100

                recall = np.sum((model_prediction * labels), axis = 1) / np.sum(labels, axis = 1)
                precision = np.sum((model_prediction * labels), axis = 1) / np.sum(model_prediction, axis = 1)
                f_score = (2*precision*recall) / (precision + recall)

                # print("recall", recall)
                # print("precision", precision)

                return test_accuracy, f_score

        def predict (self, input_vector):

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

                return prediction >= 0.5

        def get_weights (self):
                
                return self._weights

        def get_bias (self):

                return self._bias_weights
