import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import preprocess_for_gradient_descent, preprocess_for_naive_bayes
from logistic_regression import LogisticRegression
sns.set()

def plot_loss (costs, title):

        fig = plt.figure(figsize = (8,8))
        plt.plot(costs)
        plt.xticks(np.arange(0, len(costs), step = (len(costs)//20)), rotation = 45)
        plt.xlabel('Epoch')
        plt.ylabel('Cost')
        plt.title('Loss over epochs')
        plt.savefig('loss_' + title)
        plt.close(fig)


if __name__ == "__main__":


        ####################################    LOGISTIC REGRESSION   #####################################
        
        print("\n\nLOGISTIC:\n");

        data = pd.read_csv('data_banknote_authentication.txt', header = None)
        X_train, X_test, X_val, Y_train, Y_test, Y_val = preprocess_for_gradient_descent (data, 'rowwise', 'standardization', (0.7,0.15,0.15))

        # print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)
        model = LogisticRegression (learning_rate=0.005, initialisation='gaussian', regularisation='None', lambda_reg=0.01)
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
        # print(model._weights)
        # print("\n\n", model.costs)
        plot_loss (model.costs, 'logistic_5000_epochs')
