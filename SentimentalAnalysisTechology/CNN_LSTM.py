from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from Model import Model
import tensorflow.keras as K


class CNN_LSTM(Model):

    def __init__(self,*params):
        super().__init__(params)
        self.max_words_number = params[0]
        self.max_review_len = params[1]

        self.configuration = params[2]
        self.embedding_size, \
        self.kernel_size, \
        self.filters, \
        self.pool_size, \
        self.lstm_output_size, \
        self.batch_size, \
        self.epochs,\
        self.dropout,\
        self.strides,\
        self.dense = self.configuration.getConfig()

    def toJSON(self):
        pass

    def defineModel(self, x_test, y_test, x_train, y_train):

        print('Build model...')
        model = Sequential()
        model.add(Embedding(self.max_words_number, self.embedding_size, input_length=self.max_review_len))
        model.add(Dropout(self.dropout))
        model.add(Conv1D(self.filters, self.kernel_size, padding='valid', activation='relu', strides=self.strides))
        model.add(MaxPooling1D(pool_size=self.pool_size))
        model.add(LSTM(self.lstm_output_size))
        model.add(Dense(self.dense))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

        history, eval_epoch_history = self.train(self.batch_size, self.epochs, model, x_test, x_train, y_test, y_train)
        self.saveModel(model)
        return model,history, eval_epoch_history

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
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
        eval_epoch_history = model.evaluate(x_test, y_test,verbose=1)
        print('Loss:', eval_epoch_history[0])
        print('Accuracy:', "%0.2f%%" % (eval_epoch_history[1]*100))
        return history,eval_epoch_history

