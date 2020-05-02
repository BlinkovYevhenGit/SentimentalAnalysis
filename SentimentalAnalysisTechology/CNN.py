from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import time



class CNN:
    def train(self):

        print()
        print(format('How to setup a CNN model for sentiment analysis in Keras', '*^82'))

        start_time = time.time()

        # load libraries

        # load data and Set the number of words we want
        top_words = 5000
        input_length = 500

        # Load data and target vector from movie review data
        (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

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

        # Convert movie review data to feature matrix
        X_train = sequence.pad_sequences(X_train, maxlen=input_length)
        print();
        print(X_train.shape);
        print(X_train)

        X_test = sequence.pad_sequences(X_test, maxlen=input_length)
        print();
        print(X_test.shape);
        print(X_test)

        # setup a CNN network
        model = Sequential()
        model.add(Embedding(top_words, 32, input_length=input_length))
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(250, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

        # Fit the model
        model.fit(X_train, y_train, validation_data=(X_test, y_test),
                  epochs=4, batch_size=128, verbose=1)

        # Final evaluation of the model
        scores = model.evaluate(X_test, y_test, verbose=1)
        print("Accuracy: %.2f%%" % (scores[1] * 100))
        max_review_len = 500
        print();
        print("Execution Time %s seconds: " % (time.time() - start_time))
        return model
