import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

def visualize (X, T, w, discriminant_point, title = '1'):
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
                w : The direction of projection computed by the model
                discriminant_point : The discriminating point found by the model

                '''
                pos_class = X[:, np.where(T==1)[0]]
                neg_class = X[:, np.where(T==0)[0]]

                proj_pos_class = w.T.dot(pos_class)
                proj_neg_class = w.T.dot(neg_class)
                proj_mean_pos = np.mean(proj_pos_class)
                proj_mean_neg = np.mean(proj_neg_class)
                proj_stddev_pos = np.std(proj_pos_class)
                proj_stddev_neg = np.std(proj_neg_class)

                # Plot a histogram and normal distribution above the projected points, and show their intersection
                
                fig = plt.figure(figsize = (8,8))
                x1 = np.linspace(min(proj_pos_class[0]),max(proj_pos_class[0]),1000)
                x2 = np.linspace(min(proj_neg_class[0]),max(proj_neg_class[0]),1000)

                plt.hist(proj_pos_class[0],color='r', density=True, label = 'Projected Positive Class')
                plt.hist(proj_neg_class[0],color='b', density=True, label = 'Projected Negative Class')
                plt.plot(x1,stats.norm.pdf(x1, proj_mean_pos, proj_stddev_pos),color='green')
                plt.plot(x2,stats.norm.pdf(x2, proj_mean_neg, proj_stddev_neg),color='orange')

                plt.plot(discriminant_point, stats.norm.pdf(discriminant_point, proj_mean_neg, proj_stddev_neg), 's', color='black', mfc='yellow', label = 'Intersection Point')
                plt.legend(loc = "upper left")
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

                ax.scatter(proj_pos_class * w[0], proj_pos_class * w[1], color = 'r', label = "Projected Positive Class")
                ax.scatter(proj_neg_class * w[0], proj_neg_class * w[1], color = 'b', label = "Projected Negative Class")
                plt.plot(discriminant_point * w[0], discriminant_point * w[1], 's', color='black', mfc='yellow', label = "Discriminant Point")
                plt.xlabel('Feature 1')
                plt.ylabel('Feature 2')

                # To visualize the projected points in the feature 2 - feature 3 plane for a 3D dataset (Dataset 2)
                # if w.shape[0] == 3:
                #         ax.scatter(proj_pos_class * w[1], proj_pos_class * w[2], color = 'r', label = "Projected Positive Class")
                #         ax.scatter(proj_neg_class * w[1], proj_neg_class * w[2], color = 'b', label = "Projected Negative Class")
                #         plt.plot(discriminant_point * w[1], discriminant_point * w[2], 's', color='black', mfc='yellow', label = "Discriminant Point")
                #         plt.xlabel('Feature 2')
                #         plt.ylabel('Feature 3')
                
                plt.legend(loc = "upper left")
                plt.title('Projected Points : Dataset ' + title)
                plt.savefig('lda_plots/projected_points_dataset_' + title)
                plt.close (fig)

                # Visualize the normal line to the direction of projection, and the direction of projection, on the original dataset

                fig = plt.figure(figsize = (8,8))
                ax = plt.axes()

                x1 = np.linspace (-4,4,1000)

                if w.shape[0] == 2:
                        
                        # Normal to the line along which points are projected, passing through the discriminant point, in y = mx + b form
                        x2 = -(w[0]/w[1])*x1  + discriminant_point * w[1] 
                        # Actual line along which points are projected
                        x3 = (w[1]/w[0])*(x1)

                        ax.scatter(pos_class[0],pos_class[1],color='r', label = "Positive Class")
                        ax.scatter(neg_class[0],neg_class[1],color='b', label = 'Negative Class')
                        plt.plot(discriminant_point * w[0], discriminant_point * w[1], 's', color='black', mfc='yellow')

                        plt.xlabel('Feature 1')
                        plt.ylabel('Feature 2')

                # Dataset 2 has three features, so we project to plane of feature 2 and feature 3 in order to visualize
                if w.shape[0] == 3:
                        
                        x2 = -(w[1]/w[2])*x1  + discriminant_point * w[2] 
                        x3 = (w[2]/w[1])*(x1)
                        ax.scatter(pos_class[1],pos_class[2],color='r', label = "Positive Class")
                        ax.scatter(neg_class[1],neg_class[2],color='b', label = 'Negative Class')
                        plt.plot(discriminant_point * w[1], discriminant_point * w[2], 's', color='black', mfc='yellow')
                        plt.xlabel('Feature 2')
                        plt.ylabel('Feature 3')

                ax.plot(x1, x2, color='black', linewidth=2.0, label = "Discriminant Line")
                ax.plot(x1, x3, color='black', linestyle = 'dashed', linewidth=2.0, label = "Line along which points are projected")
                plt.ylim(-4,4)
                plt.legend(loc = "upper left")
                plt.title('Discriminant Line in Original Feature Space : Dataset ' + title)
                plt.savefig('lda_plots/discriminant_line_dataset_' + title)
                plt.close(fig)
