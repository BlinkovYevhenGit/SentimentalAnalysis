from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
import time
import keras as K
from Model import Model


class CNN(Model):
    def __init__(self, max_words_number, max_review_len):
        super().__init__(max_words_number, max_review_len)
        self.max_words_number = max_words_number
        self.max_review_len = max_review_len

    def toJSON(self):
        pass

    def defineModel(self, X_test, y_test, X_train, y_train, ):

        print()
        print(format('How to setup a CNN model for sentiment analysis in Keras', '*^82'))

        start_time = time.time()
        print()
        print(X_train.shape)
        print(X_train)
        print()
        print(y_train.shape)
        print(y_train)
        print()
        print(X_test.shape)
        print(X_test)
        print()
        print(y_test.shape)
        print(y_test)

        embedding_size = 32

        kernel_size = 3
        filters = 32
        pool_size = 2

        dense_units1 = 250
        dense_units2 = 1

        batch_size = 128
        epochs = 6

        # setup a CNN network
        model = Sequential()
        model.add(Embedding(self.max_words_number, embedding_size, input_length=self.max_review_len))
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=pool_size))
        model.add(Flatten())
        model.add(Dense(dense_units1, activation='relu'))
        model.add(Dense(dense_units2, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
        model.summary()

        # Fit the model

        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=1)

        # Final evaluation of the model

        eval_epoch_history = model.evaluate(X_test, y_test, verbose=1)
        print()
        # history.history['loss'].append(eval_epoch_history[0])
        # history.history['acc'].append(eval_epoch_history[1])
        print("Accuracy: %.2f%%" % (eval_epoch_history[1] * 100))
        print()
        print("Execution Time %s seconds: " % (time.time() - start_time))
        self.saveModel(model)
        return model, history,eval_epoch_history

    def runModel(self, model):
        self.doPrediction(model)

    def loadModel(self, filepath="cnn_model.h5"):
        model = K.models.load_model(".\\Models\\%s" % filepath)
        return model

    def saveModel(self, model, filename="cnn_model.h5"):
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
