import keras as K
import tensorflow as tf
from LSTM import LSTM
from CNN import CNN
from CNN_LSTM import CNN_LSTM
import numpy as np
import NaiveBayesClassifier

class Model:
    def toJSON(self):
        pass
    def runModel(self):
        #self.beginLSTM()
        #self.beginCNN()
        self.beginCNN_LSTM()
        #self.beginBayes()

    def beginBayes(self):
        NaiveBayesClassifier.launchBayesClassifier()
        # TODO: add prediction

    def beginCNN_LSTM(self):
        max_words, max_review_len, test_x, test_y, train_x, train_y = self.loadData()
        cnn_lstm_model = CNN_LSTM()
        model=cnn_lstm_model.trainCNN_LSTM(max_words,max_review_len,test_x, test_y, train_x, train_y)
        self.doPrediction(model)

    def beginCNN(self):
        max_words, max_review_len, test_x, test_y, train_x, train_y = self.loadData()
        cnn_model = CNN()
        model=cnn_model.train()
        self.doPrediction(model)

    def beginLSTM(self):
        # 0. get started
        print("\nIMDB sentiment analysis using Keras/TensorFlow ")
        np.random.seed(1)
        tf.random.set_seed(1)
        # 1. load data
        max_words,max_review_len, test_x, test_y, train_x, train_y = self.loadData()
        lstm_model = LSTM()
        model=lstm_model.defineModel(max_words, test_x, test_y, train_x, train_y)
        self.doPrediction(model)

    #TODO: add loading data for all algorithms
    def loadData(self):
        max_words = 20000
        max_review_len = 80
        print("Loading data, max unique words = %d words\n" % max_words)
        (train_x, train_y), (test_x, test_y) = \
            K.datasets.imdb.load_data(seed=1, num_words=max_words)
        train_x = K.preprocessing.sequence.pad_sequences(train_x,
                                                         truncating='pre', padding='pre',
                                                         maxlen=max_review_len)  # pad and chop!
        test_x = K.preprocessing.sequence.pad_sequences(test_x, truncating='pre', padding='pre', maxlen=max_review_len)
        return max_words,max_review_len, test_x, test_y, train_x, train_y

    def doPrediction(self,model):
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
        max_review_len = 80

        review = K.preprocessing.sequence.pad_sequences([review],
                                                        truncating='pre', padding='pre', maxlen=max_review_len)
        prediction = model.predict(review)
        print("Prediction (0 = negative, 1 = positive) = ", end="")
        print("%0.4f" % prediction[0][0])

    # TODO: add saving and loading model

    # TODO: Somehow add configuration of NNs

    # TODO: somehow create db connection

    # TODO: consider what to store in db

    # TODO: consider how to view structure of input/output data because .npz and .h5 are too complicated

