import keras as K


class LSTM:
    def train(self):
        print("Training has begun")

    def defineModel(self, max_words, test_x, test_y, train_x, train_y):
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

        bat_size = 32
        max_epochs = 4

        model = K.models.Sequential()
        model.add(K.layers.embeddings.Embedding(input_dim=max_words, output_dim=embed_vec_len, embeddings_initializer=e_init, mask_zero=mask_zero))
        model.add(K.layers.LSTM(units=units, kernel_initializer=init, dropout=dropout, recurrent_dropout=recurrent_dropout))  # 100 memory
        model.add(K.layers.Dense(units=dense_units, kernel_initializer=init, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=simple_adam, metrics=['acc'])
        print(model.summary())
        # ==================================================================
        self.trainModel(model, train_x, train_y,bat_size, max_epochs)
        # 4. evaluate model
        self.evaluateModel(model, test_x, test_y)
        # 5. save model
        self.saveModel(model)
        # 6. use model
        return model

    def saveModel(self, model):
        print("Saving model to disk \n")
        mp = ".\\Models\\imdb_model.h5"
        model.save(mp)

    def evaluateModel(self, model, test_x, test_y):
        loss_acc = model.evaluate(test_x, test_y, verbose=0)
        print("Test data: loss = %0.6f  accuracy = %0.2f%% " % \
              (loss_acc[0], loss_acc[1] * 100))

    def trainModel(self, model, train_x, train_y, bat_size, max_epochs):
        # 3. train model
        print("\nStarting training ")
        model.fit(train_x, train_y, epochs=max_epochs, batch_size=bat_size, verbose=1)
        print("Training complete \n")
