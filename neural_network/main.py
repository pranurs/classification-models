import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import preprocess_for_gradient_descent, preprocess_for_naive_bayes
from neural_network_model import Layer, NeuralNetwork
sns.set()

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

        data = pd.read_csv('data/housepricedata.csv', header = None)
        X_train, X_test, X_val, Y_train, Y_test, Y_val = preprocess_for_gradient_descent (data.iloc[1:,:], 'colwise', 'standardization')
        print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)
        
        model = NeuralNetwork(learning_rate=0.0005, initialization = 'gaussian', scaling = 'standard')
        model.add (Layer (X_train.shape[0], 'relu'))
        model.add (Layer (4, 'relu', True, 1))
        model.add (Layer (4, 'relu', True, 1))
        model.add (Layer (4, 'relu', True, 1))
        model.add (Layer (4, 'relu', True, 1))
        model.add (Layer (4, 'relu', True, 1))
        model.add (Layer (1, 'sigmoid'))
       
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
        
        