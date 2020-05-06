import keras as K

from Model import Model
class LSTM(Model):
    def __init__(self, max_words_number, max_review_len):
        super().__init__(max_words_number, max_review_len)
        self.max_words_number = max_words_number
        self.max_review_len = max_review_len

    def toJSON(self):
        pass

    def defineModel(self, test_x, test_y, train_x, train_y):
        # 2. define model
        print("Creating LSTM model")

        e_init = K.initializers.RandomUniform(-0.01, 0.01, seed=1)
        init = K.initializers.glorot_uniform(seed=1)
        simple_adam = K.optimizers.Adam()

        embed_vec_len = 32  # values per word -- 100-500 is typical
        mask_zero = True

        units = 100

        dropout = 0.2
        recurrent_dropout = 0.2
        dense_units = 1

        bat_size = 128
        max_epochs = 6

        model = K.models.Sequential()
        model.add(
            K.layers.embeddings.Embedding(input_dim=self.max_words_number, output_dim=embed_vec_len, embeddings_initializer=e_init,
                                          mask_zero=mask_zero))
        model.add(K.layers.LSTM(units=units, kernel_initializer=init, dropout=dropout,
                                recurrent_dropout=recurrent_dropout))  # 100 memory
        model.add(K.layers.Dense(units=dense_units, kernel_initializer=init, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=simple_adam, metrics=['acc'])
        print(model.summary())
        # ==================================================================
        history = self.train(model, train_x, train_y, bat_size, max_epochs)
        # 4. evaluate model
        eval_epoch_history=self.evaluateModel(model, test_x, test_y)
        # 5. save model
        # history.history['loss'].append(eval_epoch_history[0])
        # history.history['acc'].append(eval_epoch_history[1])

        self.saveModel(model)
        # 6. use model
        return model,history,eval_epoch_history

    def runModel(self, model):
        self.doPrediction(model)

    def loadModel(self, filepath="lstm_model.h5"):
        model = K.models.load_model(".\\Models\\%s" % filepath)
        return model

    def saveModel(self, model, filename="lstm_model.h5"):
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



    def evaluateModel(self, model, test_x, test_y):
        loss_acc = model.evaluate(test_x, test_y, verbose=0)
        print("Test data: loss = %0.6f  accuracy = %0.2f%% " % \
              (loss_acc[0], loss_acc[1] * 100))

        return loss_acc

    def train(self, model, train_x, train_y, bat_size, max_epochs):
        # 3. train model
        print("\nStarting training ")
        history = model.fit(train_x, train_y, epochs=max_epochs, batch_size=bat_size, verbose=1)
        print("Training complete \n")

        return history
        #print(history.history)
        # Plot history: MAE

