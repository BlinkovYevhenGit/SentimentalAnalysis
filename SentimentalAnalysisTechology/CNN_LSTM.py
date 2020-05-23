from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from Model import Model
from pymongo import MongoClient
import gridfs
import tensorflow.keras as K
from keras.callbacks import CSVLogger


from MongoManager import saveToDB, loadModelFromDB, saveConfiguration


class CNN_LSTM(Model):

    def __init__(self,*params):
        super().__init__(params)
        if len(params)==0:return
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
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client['local']
        self.fs = gridfs.GridFS(self.db)

    def toJSON(self):
        pass

    def defineModel(self, x_test, y_test, x_train, y_train):

        print('Build model...')
        model = Sequential()
        model.add(Embedding(self.max_words_number, self.embedding_size, input_length=self.max_review_len))
        #model.add(Dropout(self.dropout))
        model.add(Conv1D(self.filters, self.kernel_size, padding='same', activation='relu',))
        model.add(MaxPooling1D(pool_size=self.pool_size))
        model.add(LSTM(self.lstm_output_size, dropout=self.dropout, recurrent_dropout=self.dropout))
        model.add(Dense(self.dense, activation="sigmoid"))
        # model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

        history, eval_epoch_history = self.train(self.batch_size, self.epochs, model, x_test, x_train, y_test, y_train)
        self.saveModel(model)
        return model,history, eval_epoch_history

    def runModel(self, model,userText,review_len):
        cnn_lstm_result,prediction,definedClass  = self.doPrediction(model, userText,review_len)
        return cnn_lstm_result,prediction,definedClass

    def loadModel(self, filepath="cnn_lstm_model.h5"):
        model = loadModelFromDB(filepath)
        return model

    def saveModel(self, model, filename="cnn_lstm_model.h5"):
        saveToDB(filename, model)
        saveConfiguration("cnn_lstm_model",self.max_words_number,self.max_review_len,self.configuration.getConfig(),filename)

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

        cnn_lstm_result="Комбінована нейронна мережа(ДКЧП+ЗНМ)"
        definedClass = ""
        if prediction[0][0] >= 0.5:
            definedClass = "Позитивний"
        else:
            definedClass = "Негативний"
        return cnn_lstm_result, prediction[0][0], definedClass

    def train(self, batch_size, epochs, model, x_test, x_train, y_test, y_train):
        print('Train...')
        csv_logger = CSVLogger('cnn_lstm_log.csv',append=True,separator=";")
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), callbacks=[csv_logger])
        eval_epoch_history = model.evaluate(x_test, y_test,verbose=1)
        print('Loss:', eval_epoch_history[0])
        print('Accuracy:', "%0.2f%%" % (eval_epoch_history[1]*100))
        return history,eval_epoch_history

