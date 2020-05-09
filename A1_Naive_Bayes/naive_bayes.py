import numpy as np

class NaiveBayes:

        def __init__ (self, alpha = 1):

                self._alpha = alpha

        def fit (self, X_train, T_train, binary = 0):

                # Construct vocabulary of unique words in the training set

                vocabulary = set()

                for i in range(len(X_train)):
                        for word in X_train[i]:
                                vocabulary.add(word)

                # Construct 'Bag of Words' model for each class, get total words in each class and the class probabilities

                classes = set()
                documents_in_class = {}
                bow = {}
                word_total = {}
                class_prob = {} # Prior

                for i in range(len(T_train)):
                        classes.add(T_train[i])
                        if T_train[i] not in documents_in_class:
                                documents_in_class[T_train[i]] = []
                        documents_in_class[T_train[i]].append(X_train[i])

                for c in classes:

                        if c not in bow:
                                bow[c] = {}
                        if c not in word_total:
                                word_total[c] = 0
                        if c not in class_prob:
                                class_prob[c] = 0
                                
                        for doc in documents_in_class[c]:
                                
                                if not binary:
                                        for word in doc:
                                                if word not in bow[c]:
                                                        bow[c][word] = 0
                                                
                                                bow[c][word] += 1
                                                word_total[c] += 1
                                else:
                                        unique_words_in_doc = set(doc)
                                        for word in unique_words_in_doc:
                                                if word not in bow[c]:
                                                        bow[c][word] = 0
                                                bow[c][word] += 1
                                        
                                class_prob[c] += 1
                        
                        class_prob[c] /= len(X_train)

                # Calculating P(w|c) for each word in the vocabulary, for each class

                P_w_c = {} # Likelihood

                for word in vocabulary:
                        for c in classes:
                                if word not in bow[c]:
                                        bow[c][word] = 0
                                
                                if binary:
                                        P_w_c[(word,c)] = (bow[c][word] + self._alpha) / (len(documents_in_class[c]) + self._alpha * 2) # len(X_train))
                                else:
                                        P_w_c[(word,c)] = (bow[c][word] + self._alpha) / (word_total[c] + self._alpha * len(vocabulary))

                prediction = self.predict (X_train, class_prob, P_w_c, classes, vocabulary)
                self.train_accuracy, fscore = self.evaluate (prediction, T_train)
                print("TRAINING ACC, FSCORE: ", self.train_accuracy, fscore)

                return class_prob, P_w_c, classes, vocabulary

        def predict (self, X, prior, likelihood, classes, vocabulary):

                prediction = []

                for i in range(len(X)):
    
                        decider = {}
                        argmax = -1e6
                        predicted_class = 0
                        # unique_words = set(X[i])
                        
                        for word in X[i]:

                                if word not in vocabulary:
                                        continue

                                for c in classes:
                                        
                                        if c not in decider:
                                                decider[c] = np.log(prior[c])

                                        decider[c] += np.log(likelihood[(word,c)])

                        for c in classes:

                                if c not in decider:
                                        continue

                                if decider[c] > argmax:
                                        argmax = decider[c]
                                        predicted_class = c

                        prediction.append(predicted_class)

                return prediction

        def evaluate (self, model_prediction, labels):

                if type(labels) is list:
                        labels = np.array(labels)
                if type(model_prediction) is list:
                        model_prediction = np.array(model_prediction)

                labels = np.reshape(labels, (labels.shape[0], -1))
                model_prediction = np.reshape(model_prediction, (model_prediction.shape[0], -1))

                test_accuracy = np.sum((model_prediction - labels)**2, axis = 0)
                test_accuracy = (1 - test_accuracy / model_prediction.shape[0]) * 100

                recall = np.sum((model_prediction * labels), axis = 0) / np.sum(labels, axis = 0)
                precision = np.sum((model_prediction * labels), axis = 0) / np.sum(model_prediction, axis = 0)
                f_score = (2*precision*recall) / (precision + recall)

                # print("recall", recall)
                # print("precision", precision)

                return test_accuracy, f_score

