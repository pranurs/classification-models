import numpy as np
import scipy.stats as stats

def solve (mu1, mu2, sigma1, sigma2):
        '''
        Utility function to return the intersection point of 2 normal distributions

        Parameters:
        mu1 : Mean of the first normal distribution
        mu2 : Mean of the second normal distribution
        sigma1 : Standard deviation of the first normal distribution
        sigma2 : Standard deviation of the second normal distribution

        Returns:
        The roots obtained on solving the quadratic equation formed when equating the two normal distributions.
        They represent the intersection points of the two normal distributions.
        '''
        a = sigma1**2 - sigma2**2
        b = -2*(mu2 * sigma1**2 - mu1 * sigma2**2)
        c = (sigma1**2 * mu2**2 - sigma2**2 * mu1**2) - (2 * sigma1**2 * sigma2**2 * np.log(sigma1/sigma2))
        
        return np.roots([a,b,c])

class LDA:
        '''
        A class for Linear Discriminant Analysis.

        Attributes:

        _w : Optimal direction of projection of the data points
        _discriminant_point : Final discriminant point separating the two classes after projection along _w
        _orientation_after_collapse : Captures which side of the discriminant point is representative of the positive class (for visualization and predictive purposes)

        Methods:

        fit (X, T) : Updates the instance variables _w, _discriminant_point, _orientation_after_collapse for the passed data examples and labels
        predict (X) : Returns a vector containing the predicted class label for each data example using the instance variables
        evaluate (model_prediction, labels) : Returns the test accuracy and f1-score of the final prediction made by the model instance.
        get_w () : Returns the the direction of projection computed
        get_discriminant_point () : Returns the discriminant point computed
        '''

        def __init__ (self, w = None, discriminant_point = None, orientation_after_collapse = None):
                '''
                Initializes the instance variables:
                        optimal direction of projection (_w)
                        discriminant point after projection (_discriminant_point)
                        orientation after collapse (_orientation_after_collapse)

                If no values passed, initializes to None

                Optional parameters:
                w : ndarray, defaults to None
                discriminant_point : float, defaults to None
                orientation_after_collapse : tuple, defaults to None
                '''

                self._w = w
                self._discriminant_point = discriminant_point
                self._orientation_after_collapse = orientation_after_collapse

        def fit (self, X, T):
                '''
                Computes the optimal direction of projection and the discriminant point for the passed data points.
                Updates the instance variables accordingly.

                Parameters:
                X : An array of data examples, wherein each column is an individual data instance (feature vector)
                T : A rank one column vector containing labels
                '''
                
                # Separating instances into positive class and negative class
                pos_class = X[:, np.where(T==1)[0]]
                neg_class = X[:, np.where(T==0)[0]]

                # Calculating mean of positive and negative classes independently
                mean_pos = np.sum(pos_class, axis = 1) / pos_class.shape[1]
                mean_neg = np.sum(neg_class, axis = 1) / neg_class.shape[1]
                mean_pos = mean_pos.reshape((mean_pos.shape[0], 1))
                mean_neg = mean_neg.reshape((mean_neg.shape[0], 1))

                # Calculating number of positive instances and negative instances
                n_pos = pos_class.shape[1]
                n_neg = neg_class.shape[1]

                # Calculating the within-class covariance matrix
                Sw = ((pos_class - mean_pos).dot((pos_class - mean_pos).T) / n_pos) + ((neg_class - mean_neg).dot((neg_class - mean_neg).T) / n_neg)

                # Calculating the between-class covariance matrix
                diff_means = mean_pos - mean_neg
                Sb = diff_means.dot(diff_means.T)

                # Finding the eigenvectors and corresponding eigenvalues for inverse(Sw)*Sb
                A = np.linalg.inv(Sw).dot(Sb)
                eigenvalues, eigenvectors = np.linalg.eig(A)

                # Sorting the eigenvectors in descending order of the magnitudes of their eigenvalues
                pairs = [(np.abs(eigenvalues[i]), eigenvectors[:,i]) for i in range(len(eigenvalues))]
                pairs = sorted(pairs, key = lambda x:x[0], reverse = True)

                # The direction of projection (w) (for 1D) is the eigenvector with maximum eigenvalue
                w = pairs[0][1]
                w = w.T # Make w a column matrix
                w = w.reshape((w.shape[0], 1))

                # Calculating the projected parameters for the instances which have been fit
                proj_pos_class = w.T.dot(pos_class)
                proj_neg_class = w.T.dot(neg_class)
                proj_mean_pos = np.mean(proj_pos_class)
                proj_mean_neg = np.mean(proj_neg_class)
                proj_stddev_pos = np.std(proj_pos_class)
                proj_stddev_neg = np.std(proj_neg_class)
                intersection_point = solve (proj_mean_pos, proj_mean_neg, proj_stddev_pos, proj_stddev_neg)
                intersection_point = [x for x in intersection_point if (x > min(proj_mean_neg,proj_mean_pos)) and (x < max(proj_mean_neg,proj_mean_pos))]

                self._w = w
                self._discriminant_point = intersection_point

                if proj_mean_pos < intersection_point:
                        self._orientation_after_collapse = (1,0)
                else:
                        self._orientation_after_collapse = (0,1)

        def predict (self, X):
                '''
                Predicts the class of the passed data points, using the instance variables for direction of projection, 
                discriminant point, and orientation after collapse.

                Parameters:
                X : An array of data examples for which prediction is to be done, wherein each column is an individual data instance (feature vector)

                Returns:
                prediction : A vector containing the final predicted class (1 or 0) for each data example
                '''

                # X must have test instances (feature vectors) as columns
                proj_X = self._w.T.dot(X)

                if self._orientation_after_collapse[0] == 1:
                        prediction = proj_X < self._discriminant_point
                else:
                        prediction = proj_X > self._discriminant_point

                return prediction

        def evaluate (self, model_prediction, labels):
                '''
                Returns the test accuracy and f1-score of the final prediction made by the model instance.

                Parameters:
                model_prediction : Vector containing class predictions for each data example
                labels : Corresponding vector containing the correct class labels for each data example.

                Returns:
                test_accuracy : The accuracy of the prediction
                f_score : The f1-score of the prediction
                '''

                labels = labels.reshape((labels.shape[0], -1))
                labels = labels.T

                test_accuracy = np.sum((model_prediction - labels)**2, axis = 1)
                test_accuracy = (1 - test_accuracy / model_prediction.shape[1]) * 100

                recall = np.sum((model_prediction * labels), axis = 1) / np.sum(labels, axis = 1)
                precision = np.sum((model_prediction * labels), axis = 1) / np.sum(model_prediction, axis = 1)
                f_score = (2*precision*recall) / (precision + recall)

                # print("recall", recall)
                # print("precision", precision)

                return test_accuracy, f_score

        def get_w (self):

                return self._w

        def get_discriminant_point (self):

                return self._discriminant_point