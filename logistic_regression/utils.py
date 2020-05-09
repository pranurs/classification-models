import numpy as np

def sigmoid (z):
        return 1/(1 + np.exp(-z));

def tanh (z):
        return (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))

def relu (z):
        return z * (z > 0)

def leaky_relu (z):
        return np.maximum (0.01*z, z)

def activate (activation, z):

        if activation == 'sigmoid':
                return sigmoid(z)

        elif activation == 'tanh':
                return tanh(z)

        elif activation == 'relu':
                return relu(z)

        elif activation == 'leaky_relu':
                return leaky_relu(z)

        elif activation == 'identity':
                return z
        
        else:
                raise Exception ('Invalid activation')

def calc_der (activation, z):

        if activation == 'sigmoid':
                return sigmoid(z) * (1 - sigmoid(z))

        elif activation == 'tanh':
                return 1 - (tanh(z))**2

        elif activation == 'relu':
                return z >= 0

        elif activation == 'leaky_relu':
                return np.maximum (0.01, z >= 0)

        elif activation == 'identity':
                return 1
        
        else:
                raise Exception ('Invalid activation')

if __name__ == '__main__':

        for i in range(-100,100):
                print(i, "leaky", leaky_relu(i))
                # print(i, "tanh", tanh(i))