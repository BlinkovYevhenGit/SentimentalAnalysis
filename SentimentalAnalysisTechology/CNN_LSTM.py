from __future__ import print_function
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.datasets import imdb
import keras as K


class CNN_LSTM:
    def trainCNN_LSTM(self, max_features, maxlen,x_test, y_test, x_train, y_train):

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
        epochs = 2

        '''
        Note:
        batch_size is highly sensitive.
        Only 2 epochs are needed as the dataset is very small.
        '''
        print('Build model...')

        model = Sequential()
        model.add(Embedding(max_features, embedding_size, input_length=maxlen))
        model.add(Dropout(0.25))
        model.add(Conv1D(filters,
                         kernel_size,
                         padding='valid',
                         activation='relu',
                         strides=1))
        model.add(MaxPooling1D(pool_size=pool_size))
        model.add(LSTM(lstm_output_size))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        self.train(batch_size, epochs, model, x_test, x_train, y_test, y_train)
        return model
        # # 5. save model
        # print("Saving model to disk \n")
        # mp = ".\\Models\\imdb_model.h5"
        # model.save(mp)
        # 6. use model

    def train(self, batch_size, epochs, model, x_test, x_train, y_test, y_train):
        print('Train...')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test))
        score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
        print('Test score:', score)
        print('Test accuracy:', acc)

