import math
import time
from collections import defaultdict

import numpy as np
from Model import Model

class NaiveBayesClassifier(Model):
    def __init__(self,*params):
        super().__init__(params)
        self.n_gram = 1,
        self.prior = defaultdict(int)
        self.logprior = {}
        self.bigdoc = defaultdict(list)
        self.loglikelihoods = defaultdict(defaultdict)
        self.V = []
        self.n = self.n_gram

    def toJSON(self):
        pass

    def defineModel(self, test_x, test_y, train_x, train_y):
        start = time.time()
        self.train(train_x, train_y, alpha=1)
        results, acc,bayes_result,classTone,definedClass = self.evaluate_predictions(test_x, test_y, verbose=0)
        end = time.time()
        print('Ran in {} seconds'.format(round(end - start, 3)))
        return self,acc

    def runModel(self, model,userText, review_len):
        print("Testing review - "+userText)
        validation_set = [userText]
        validation_labels = [1]
        results, acc,bayes_result,classTone,definedClass=model.evaluate_predictions(validation_set, validation_labels, verbose=1)
        return results, acc,bayes_result,classTone,definedClass
    def loadModel(self,filename):
        pass

    def saveModel(self, model):
        pass

    def compute_vocabulary(self, documents):
        vocabulary = set()

        for doc in documents:
            for word in doc.split(" "):
                vocabulary.add(word.lower())

        return vocabulary

    def count_word_in_classes(self):
        counts = {}
        for c in list(self.bigdoc.keys()):
            docs = self.bigdoc[c]
            counts[c] = defaultdict(int)
            for doc in docs:
                words = doc.split(" ")
                for word in words:
                    counts[c][word] += 1

        return counts

    def train(self, training_set, training_labels, alpha=1):
        # Get number of documents
        N_doc = len(training_set)

        # Get vocabulary used in training set
        self.V = self.compute_vocabulary(training_set)

        # Create bigdoc
        for x, y in zip(training_set, training_labels):
            self.bigdoc[y].append(x)

        # Get set of all classes
        all_classes = set(training_labels)

        # Compute a dictionary with all word counts for each class
        self.word_count = self.count_word_in_classes()

        # For each class
        for c in all_classes:
            # Get number of documents for that class
            N_c = float(sum(training_labels == c))

            # Compute logprior for class
            self.logprior[c] = np.log(N_c / N_doc)

            # Calculate the sum of counts of words in current class
            total_count = 0
            for word in self.V:
                total_count += self.word_count[c][word]

            # For every word, get the count and compute the log-likelihood for this class
            for word in self.V:
                count = self.word_count[c][word]
                self.loglikelihoods[c][word] = np.log((count + alpha) / (total_count + alpha * len(self.V)))

    def predict(self, test_doc):
        sums = {
            0: 0,
            1: 0,
        }
        for c in self.bigdoc.keys():
            sums[c] = self.logprior[c]
            words = test_doc.split(" ")
            for word in words:
                if word in self.V:
                    sums[c] += self.loglikelihoods[c][word]

        return sums


    def evaluate_predictions(self,validation_set, validation_labels, verbose):
        correct_predictions = 0
        predictions_list = []
        prediction = -1
        bayes_result=""
        class1=0.0
        class2=0.0
        definedClass = ""
        classTone = 0.0
        for dataset, label in zip(validation_set, validation_labels):
            probabilities = self.predict(dataset)
            if verbose == 1:
                print(probabilities)
                class1 = 1 / (1 + (math.exp(probabilities[1] - probabilities[0])))
                class2 = 1 / (1 + (math.exp(probabilities[0] - probabilities[1])))
                print("Class probability", max(class1, class2) * 100, "%")
                bayes_result = "Наївний Баєсів Класифікатор"

            if probabilities[0] >= probabilities[1]:
                prediction = 0
                definedClass = "Негативний"
                classTone=class1
            elif probabilities[0] < probabilities[1]:
                prediction = 1
                definedClass = "Позитивний"
                classTone=class2
            if prediction == label:
                correct_predictions += 1
                predictions_list.append("+")
            else:
                predictions_list.append("-")

        print("Predicted correctly {} out of {} ({}%)".format(correct_predictions, len(validation_labels), round(correct_predictions / len(validation_labels) * 100, 5)))
        return predictions_list, round(correct_predictions / len(validation_labels) * 100),bayes_result,classTone,definedClass


