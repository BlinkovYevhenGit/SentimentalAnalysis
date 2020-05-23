from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Embedding
import time
import tensorflow.keras as K
from Model import Model

from MongoManager import saveToDB, loadModelFromDB, saveConfiguration


class CNN(Model):
    def __init__(self, *params):
        super().__init__(params)
        if len(params)==0:return
        self.max_words_number = params[0]
        self.max_review_len = params[1]

        self.configuration = params[2]
        self.embedding_size, \
        self.kernel_size, \
        self.filters, \
        self.pool_size, \
        self.dense_units1, \
        self.dense_units2, \
        self.batch_size, \
        self.epochs = self.configuration.getConfig()


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

        # setup a CNN network
        model = Sequential()
        model.add(Embedding(self.max_words_number, self.embedding_size, input_length=self.max_review_len))
        model.add(Conv1D(filters=self.filters, kernel_size=self.kernel_size, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=self.pool_size))
        model.add(Flatten())
        model.add(Dense(self.dense_units1, activation='relu'))
        model.add(Dense(self.dense_units2, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
        model.summary()

        # Fit the model

        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=self.epochs, batch_size=self.batch_size, verbose=1)

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

    def runModel(self, model,userText,review_len):
        cnn_result,prediction,definedClass = self.doPrediction(model, userText,review_len)
        return cnn_result, prediction, definedClass

    def loadModel(self, filepath="cnn_model.h5"):
        model = loadModelFromDB(filepath)

        return model

    def saveModel(self, model, filename="cnn_model.h5"):
        saveToDB(filename, model)
        saveConfiguration("cnn_model",self.max_words_number,self.max_review_len,self.configuration.getConfig(),filename)

    def doPrediction(self, model, userText, max_review_len):
        print("New review:" + userText)
        d = K.datasets.imdb.get_word_index()
        review = userText
        words = review.split()
        review = []
        for word in words:
            if word not in d:
                review.append(2)
            else:
                review.append(d[word] + 3)

        review = K.preprocessing.sequence.pad_sequences([review], truncating='pre', padding='pre',
                                                        maxlen=max_review_len)
        prediction = model.predict(review)
        print("Prediction (0 = negative, 1 = positive) = ", end="")
        print("%0.4f" % prediction[0][0])
        cnn_result = "Згорткова нейронна мережа"
        definedClass=""
        if prediction[0][0]>=0.5 : definedClass="Позитивний"
        else:definedClass="Негативний"
        return cnn_result,prediction[0][0],definedClass
