import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
sns.set()

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
        visualize (X, T) : Renders plots to visualize the final results after projecting data points, using the instance variables of the model.
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

        def visualize (self, X, T, title = '1'):
                '''
                Renders plots to visualize the final results after projecting data points, using the instance variables of the model.
                        1. Histogram and normal distribution plotted above the projected points, intersection shown
                        2. Points plotted after projection, in original feature space
                        3. Direction of projection and discriminant line plotted in original feature space
                
                Plots rendered are saved in ./lda_plots/

                Optional Parameters:
                title : suffix in title used to save plot, defaults to "1"

                Necessary Parameters:
                X : An array of data examples, wherein each column is an individual data instance (feature vector)
                T : A rank one column vector containing labels
                '''

                pos_class = X[:, np.where(T==1)[0]]
                neg_class = X[:, np.where(T==0)[0]]

                proj_pos_class = self._w.T.dot(pos_class)
                proj_neg_class = self._w.T.dot(neg_class)
                proj_mean_pos = np.mean(proj_pos_class)
                proj_mean_neg = np.mean(proj_neg_class)
                proj_stddev_pos = np.std(proj_pos_class)
                proj_stddev_neg = np.std(proj_neg_class)

                # Plot a histogram and normal distribution above the projected points, and show their intersection
                
                fig = plt.figure(figsize = (8,8))
                x1 = np.linspace(min(proj_pos_class[0]),max(proj_pos_class[0]),1000)
                x2 = np.linspace(min(proj_neg_class[0]),max(proj_neg_class[0]),1000)

                plt.hist(proj_pos_class[0],color='r',density=True)
                plt.hist(proj_neg_class[0],color='b',density=True)
                plt.plot(x1,stats.norm.pdf(x1, proj_mean_pos, proj_stddev_pos),color='green')
                plt.plot(x2,stats.norm.pdf(x2, proj_mean_neg, proj_stddev_neg),color='orange')

                intersection_point = solve (proj_mean_pos, proj_mean_neg, proj_stddev_pos, proj_stddev_neg)
                intersection_point = [x for x in intersection_point if (x > min(proj_mean_neg,proj_mean_pos)) and (x < max(proj_mean_neg,proj_mean_pos))]
                plt.plot(intersection_point, stats.norm.pdf(intersection_point, proj_mean_neg, proj_stddev_neg), 's', color='black', mfc='yellow')
                plt.ylabel('Number of points')
                plt.xlabel('Projected values along 1D')
                plt.title('Normal Distribution for Projected Points : Dataset ' + title)
                plt.savefig('lda_plots/normal_dist_plot_dataset_' + title)
                plt.close(fig)

                # Plot the points after projection

                fig = plt.figure(figsize = (8,8))
                ax = plt.axes()

                # Slope of line along which they are projected, in original feature space = w[1]/w[0] (if feature space > 3D, this projects it to a 2D plane)
                # projected class values = distance along the line of projection from the origin (say) = dist
                # Hence, slope = tan(theta) = w[1]/w[0] and as w[0]^2 + w[1]^2 = 1, cos(theta) = w[0] and sin(theta) = w[1]
                # x coordinate in original feature space = dist*cos(theta)
                # y coordinate in original feature space = dist*sin(theta)

                ax.scatter(proj_neg_class * self._w[0], proj_neg_class * self._w[1], color = 'b')
                ax.scatter(proj_pos_class * self._w[0], proj_pos_class * self._w[1], color = 'r')
                plt.plot(intersection_point * self._w[0], intersection_point * self._w[1], 's', color='black', mfc='yellow')
                plt.xlabel('Feature 1')
                plt.ylabel('Feature 2')

                # To visualize the projected points in the feature 2 - feature 3 plane for a 3D dataset (Dataset 2)
                # if self._w.shape[0] == 3:
                #         ax.scatter(proj_neg_class * self._w[1], proj_neg_class * self._w[2], color = 'b')
                #         ax.scatter(proj_pos_class * self._w[1], proj_pos_class * self._w[2], color = 'r')
                #         plt.plot(intersection_point * self._w[1], intersection_point * self._w[2], 's', color='black', mfc='yellow')
                #         plt.xlabel('Feature 2')
                #         plt.ylabel('Feature 3')
                        
                plt.title('Projected Points : Dataset ' + title)
                plt.savefig('lda_plots/projected_points_dataset_' + title)

                plt.close (fig)

                # Visualize the normal line to the direction of projection, and the direction of projection, on the original dataset

                fig = plt.figure(figsize = (8,8))
                ax = plt.axes()

                x1 = np.linspace (-4,4,1000)

                if self._w.shape[0] == 2:
                        
                        # Normal to the line along which points are projected, passing through the discriminant point, in y = mx + b form
                        x2 = -(self._w[0]/self._w[1])*x1  + intersection_point * self._w[1] 
                        # Actual line along which points are projected
                        x3 = (self._w[1]/self._w[0])*(x1)

                        ax.scatter(pos_class[0],pos_class[1],color='r')
                        ax.scatter(neg_class[0],neg_class[1],color='b')
                        plt.plot(intersection_point * self._w[0], intersection_point * self._w[1], 's', color='black', mfc='yellow')

                        plt.xlabel('Feature 1')
                        plt.ylabel('Feature 2')

                # Dataset 2 has three features, so we project to plane of feature 2 and feature 3 in order to visualize
                if self._w.shape[0] == 3:
                        
                        x2 = -(self._w[1]/self._w[2])*x1  + intersection_point * self._w[2] 
                        x3 = (self._w[2]/self._w[1])*(x1)
                        ax.scatter(pos_class[1],pos_class[2],color='r')
                        ax.scatter(neg_class[1],neg_class[2],color='b')
                        plt.plot(intersection_point * self._w[1], intersection_point * self._w[2], 's', color='black', mfc='yellow')
                        plt.xlabel('Feature 2')
                        plt.ylabel('Feature 3')

                ax.plot(x1, x2, color='black', linestyle = 'dashed', linewidth=2.0)
                ax.plot(x1, x3, color='black', linewidth=2.0)
                plt.ylim(-4,4)
                
                plt.title('Discriminant Line in Original Feature Space : Dataset ' + title)
                plt.savefig('lda_plots/discriminant_line_dataset_' + title)
                plt.close(fig)
