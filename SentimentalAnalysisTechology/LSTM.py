from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding
import tensorflow.keras as K
from keras.callbacks import CSVLogger
from pymongo import MongoClient
import gridfs
import os
from LSTM_Config import LSTMConfiguration
from Model import Model
from MongoManager import saveToDB, loadModelFromDB, saveConfiguration


class LSTM(Model):
    def __init__(self, *params):
        super().__init__(params)
        self.max_words_number = params[0]
        self.max_review_len = params[1]
        self.configuration=params[2]
        self.embed_vec_len,\
        self.mask_zero,\
        self.units,\
        self.dropout,\
        self.recurrent_dropout,\
        self.dense_units,\
        self.bat_size,\
        self.max_epochs = self.configuration.getConfig()

        #32  # values per word -- 100-500 is typical
        # self.mask_zero = params[3][3]True
        #
        # self.units = params[3][3]100
        #
        # self.dropout = params[3][4]0.2
        # self.recurrent_dropout = params[3][5]0.2
        # self.dense_units = params[3][]61
        #
        # self.bat_size = params[3][7]128
        # self.max_epochs = params[3][8]6

    def toJSON(self):
        pass

    def defineModel(self, test_x, test_y, train_x, train_y):
        # 2. define model
        print("Creating LSTM model")

        e_init = K.initializers.RandomUniform(-0.01, 0.01, seed=1)
        init = K.initializers.glorot_uniform(seed=1)
        simple_adam = K.optimizers.Adam()



        model = Sequential()
        model.add(Embedding(input_dim=self.max_words_number, output_dim=self.embed_vec_len, embeddings_initializer=e_init, mask_zero=self.mask_zero))
        model.add(K.layers.LSTM(units=self.units, kernel_initializer=init, dropout=self.dropout,
                                recurrent_dropout=self.recurrent_dropout))  # 100 memory
        model.add(Dense(units=self.dense_units, kernel_initializer=init, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=simple_adam, metrics=['acc'])
        print(model.summary())
        # ==================================================================
        history = self.train(model, train_x, train_y, self.bat_size, self.max_epochs)
        # 4. evaluate model
        eval_epoch_history=self.evaluateModel(model, test_x, test_y)
        # 5. save model
        # history.history['loss'].append(eval_epoch_history[0])
        # history.history['acc'].append(eval_epoch_history[1])

        self.saveModel(model)
        # 6. use model
        return model,history,eval_epoch_history

    def runModel(self, model,userText,review_len):
        lstm_result =self.doPrediction(model,userText,review_len)
        return lstm_result

    def loadModel(self, filepath="lstm_model.h5"):
        model = loadModelFromDB(filepath)
        return model

    def saveModel(self, model, filename="lstm_model.h5"):
        saveToDB(filename, model)
        saveConfiguration(filename.replace(".h5",""),self.max_words_number,self.max_review_len,self.configuration.getConfig(),filename)

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
        lstm_result = "Рекурентна нейронна мережа з довгою короткочасною пам'яттю - Прогноз: (0 = негативний, 1 = позитивний) = %0.4f" % \
                     prediction[0][0]
        return lstm_result




    def evaluateModel(self, model, test_x, test_y):
        loss_acc = model.evaluate(test_x, test_y, verbose=1)
        print("Test data: loss = %0.6f  accuracy = %0.2f%% " % \
              (loss_acc[0], loss_acc[1] * 100))

        return loss_acc

    def train(self, model, train_x, train_y, bat_size, max_epochs):
        # 3. train model
        print("\nStarting training ")
        csv_logger = CSVLogger('lstm_log.csv',append=True,separator=";")
        history = model.fit(train_x, train_y, epochs=max_epochs, batch_size=bat_size, verbose=1, callbacks=[csv_logger])
        print("Training complete \n")

        return history
        #print(history.history)
        # Plot history: MAE

