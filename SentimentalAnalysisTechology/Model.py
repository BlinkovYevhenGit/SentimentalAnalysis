import keras as K

from DataLoader import loadData, loadBayesData
from LSTM import LSTM
from CNN import CNN
from CNN_LSTM import CNN_LSTM
import NaiveBayesClassifier

from NaiveBayesClassifier import evaluate_predictions


class Model:
    def __init__(self):
        self.max_words_number = 2000
        self.max_review_len = 100

    def toJSON(self):
        pass

    def runModel(self):
        self.beginLSTM()
        # self.beginCNN()
         #self.beginCNN_LSTM()
        #self.beginBayes()

    def beginBayes(self):
        training_set, training_labels, validation_set, validation_labels = loadBayesData()
        NBClassifier = NaiveBayesClassifier.launchBayesClassifier(training_set, training_labels, validation_set, validation_labels)
        self.predictBayes(NBClassifier)

    def beginCNN_LSTM(self):
        test_x, test_y, train_x, train_y = loadData(self.max_words_number, self.max_review_len)
        cnn_lstm_model = CNN_LSTM()
        model = cnn_lstm_model.trainCNN_LSTM(self.max_words_number, self.max_review_len, test_x, test_y, train_x, train_y)
        self.doPrediction(model)

    def beginCNN(self):
        test_x, test_y, train_x, train_y = loadData(self.max_words_number, self.max_review_len)
        cnn_model = CNN()
        model = cnn_model.train(test_x, test_y, train_x, train_y, self.max_words_number, self.max_review_len)
        self.doPrediction(model)

    def beginLSTM(self):
        # 0. get started
        print("\nIMDB sentiment analysis using Keras/TensorFlow ")
        # 1. load data
        test_x, test_y, train_x, train_y = loadData(self.max_words_number, self.max_review_len)
        lstm_model = LSTM()
        model = lstm_model.defineModel(self.max_words_number, test_x, test_y, train_x, train_y)
        self.doPrediction(model)

    # TODO: add loading data for all algorithms

    def doPrediction(self, model):
        print("New review: \'The movie was awesome. I love it \'")
        d = K.datasets.imdb.get_word_index()
        review = "The movie was awesome. I love it"
        words = review.split()
        review = []
        for word in words:
            if word not in d:
                review.append(2)
            else:
                review.append(d[word] + 3)

        review = K.preprocessing.sequence.pad_sequences([review], truncating='pre', padding='pre', maxlen=self.max_review_len)
        prediction = model.predict(review)
        print("Prediction (0 = negative, 1 = positive) = ", end="")
        print("%0.4f" % prediction[0][0])

    def predictBayes(self, NBclassifier):
        print("Testing review - The movie was awesome. I love it")
        validation_set = ["The movie was awesome. I love it"]
        validation_labels = [1]
        evaluate_predictions(validation_set, validation_labels, NBclassifier, verbose=1)

    # TODO: add saving and loading model

    # TODO: Somehow add configuration of NNs

    # TODO: somehow create db connection

    # TODO: consider what to store in db

    # TODO: consider how to view structure of input/output data because .npz and .h5 are too complicated



