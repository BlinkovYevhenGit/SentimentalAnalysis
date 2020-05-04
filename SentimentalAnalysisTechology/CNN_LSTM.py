from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from Model import Model
import keras as K


class CNN_LSTM(Model):

    def __init__(self, max_words_number, max_review_len):
        super().__init__(max_words_number, max_review_len)
        self.max_words_number = max_words_number
        self.max_review_len = max_review_len

    def toJSON(self):
        pass

    def defineModel(self, x_test, y_test, x_train, y_train):
        # Embedding
        embedding_size = 128

        # Convolution
        kernel_size = 5
        filters = 64
        pool_size = 4

        # LSTM
        lstm_output_size = 70

        # Training
        batch_size = 30
        epochs = 4

        dropout = 0.25
        strides = 1
        dense = 1

        '''
        Note:
        batch_size is highly sensitive.
        Only 2 epochs are needed as the dataset is very small.
        '''
        print('Build model...')

        model = Sequential()
        model.add(Embedding(self.max_words_number, embedding_size, input_length=self.max_review_len))
        model.add(Dropout(dropout))
        model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=strides))
        model.add(MaxPooling1D(pool_size=pool_size))
        model.add(LSTM(lstm_output_size))
        model.add(Dense(dense))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.train(batch_size, epochs, model, x_test, x_train, y_test, y_train)
        self.saveModel(model)
        return model

    def runModel(self, model):
        self.doPrediction(model)

    def loadModel(self, filepath="cnn_lstm_model.h5"):
        model = K.models.load_model(".\\Models\\%s" % filepath)
        return model

    def saveModel(self, model, filename="cnn_lstm_model.h5"):
        print("Saving model to disk \n")
        mp = ".\\Models\\" + filename
        model.save(mp)

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

        review = K.preprocessing.sequence.pad_sequences([review], truncating='pre', padding='pre',
                                                        maxlen=self.max_review_len)
        prediction = model.predict(review)
        print("Prediction (0 = negative, 1 = positive) = ", end="")
        print("%0.4f" % prediction[0][0])

    def train(self, batch_size, epochs, model, x_test, x_train, y_test, y_train):
        print('Train...')
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
        score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
        print('Test score:', score)
        print('Test accuracy:', acc)

