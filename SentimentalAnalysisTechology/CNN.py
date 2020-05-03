from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import time



class CNN:
    def train(self,X_train,y_train,X_test,y_test, top_words,input_length):

        print()
        print(format('How to setup a CNN model for sentiment analysis in Keras', '*^82'))

        start_time = time.time()
        print();
        print(X_train.shape);
        print(X_train)
        print();
        print(y_train.shape);
        print(y_train)
        print();
        print(X_test.shape);
        print(X_test)
        print();
        print(y_test.shape);
        print(y_test)

        embedding_size = 32

        kernel_size = 3
        filters = 32
        pool_size = 2

        dense_units1 = 250
        dense_units2 = 250

        batch_size = 128
        epochs = 4

        # setup a CNN network
        model = Sequential()
        model.add(Embedding(top_words, embedding_size, input_length=input_length))
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=pool_size))
        model.add(Flatten())
        model.add(Dense(dense_units1, activation='relu'))
        model.add(Dense(dense_units2, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

        # Fit the model

        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=1)

        # Final evaluation of the model
        scores = model.evaluate(X_test, y_test, verbose=1)
        print("Accuracy: %.2f%%" % (scores[1] * 100))
        print();
        print("Execution Time %s seconds: " % (time.time() - start_time))
        return model
