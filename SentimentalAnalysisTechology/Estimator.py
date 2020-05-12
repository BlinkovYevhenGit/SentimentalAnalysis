from LSTM import LSTM
from CNN import CNN
from CNN_LSTM import CNN_LSTM
from NaiveBayesClassifier import NaiveBayesClassifier
from DataLoader import loadData
from DataLoader import loadBayesData
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from LSTM_Config import LSTMConfiguration
from CNN_Config import CNNConfiguration
from CNN_LSTM_Config import CNN_LSTMConfiguration


class Estimator:
    def estimate(self, top_words,max_words_number,LSTM_Config, CNN_Config,CNN_LSTM_Config, userText):
        accList = dict()
        lossList = dict()

        histories = list()

        bayesResults = self.runBayes(top_words,max_words_number,userText)
        accList.update({"Naive Bayes": bayesResults / 100})

        historyCNN_LSTM, evalCNN_LSTM = self.runCNN_LSTM(top_words,max_words_number,CNN_LSTM_Config,userText)

        histories.append(tuple(("CNN_LSTM", historyCNN_LSTM)))
        lossList.update({"CNN_LSTM": evalCNN_LSTM[0]})
        accList.update({"CNN_LSTM": evalCNN_LSTM[1]})

        historyCNN, evalCNN =  self.runCNN(top_words,max_words_number,CNN_Config,userText)

        histories.append(tuple(("CNN", historyCNN)))
        lossList.update({"CNN": evalCNN[0]})
        accList.update({"CNN": evalCNN[1]})

        historyLSTM, evalLSTM = self.runLSTM(top_words,max_words_number,LSTM_Config,userText)

        histories.append(tuple(("LSTM", historyLSTM)))
        lossList.update({"LSTM": evalLSTM[0]})
        accList.update({"LSTM": evalLSTM[1]})

        plot = self.outputGraph(histories)
        self.showTestResultsGraph(accList, lossList, plot)
        return histories,lossList,accList

    def showTestResultsGraph(self, accList, lossList, plot):
        accNames = list(accList.keys())
        accValues = list(accList.values())
        lossNames = list(lossList.keys())
        lossValues = list(lossList.values())
        plot.figure(1)
        fig, axs = plot.subplots(1, 2, figsize=(13, 5))
        axs[0].bar(accNames, accValues)
        axs[0].set_title("Точність аналізу тексту з тестової вибірки")
        axs[1].bar(lossNames, lossValues)
        axs[1].set_title("Значення функції втрат")
        fig.suptitle('Ефективність алгоритмів на текстах з тестової вибірки', y=0.99)
        plot.savefig("./static/graph1.png")

    def outputGraph(self, histories):
        plt.figure(figsize=(13, 5))
        for history in histories:
            algoName, algoHistory = history
            plt.plot(algoHistory.history['loss'], label=('%s training loss' % algoName))
            plt.plot(algoHistory.history['acc'], label=('%s training accuracy' % algoName))
            plt.legend(loc="upper left")
        plt.xlabel('№ епохи навчання')
        plt.ylabel('Значення')
        plt.title('Ефективність алгоритмів впродовж періоду тренування моделей', y=0.99)
        plt.savefig("./static/graph2.png")
        return plt

    def runLSTM(self,top_words,max_words_number,LSTM_Config,userText):

        #config = LSTMConfiguration(32, True, 100, 0.2, 0.2, 1, 128, 3)
        config = LSTMConfiguration(LSTM_Config[0],LSTM_Config[1],LSTM_Config[2],LSTM_Config[3],
                                   LSTM_Config[4],LSTM_Config[5],LSTM_Config[6],LSTM_Config[7])

        #lstm = LSTM(5000, 50, config)
        lstm = LSTM(top_words,max_words_number, config)

        test_x, test_y, train_x, train_y = loadData(lstm.max_words_number, lstm.max_review_len)

        model, history, eval_epoch_history = lstm.defineModel(test_x, test_y, train_x, train_y)

        # model = lstm.loadModel()

        lstm.runModel(model,userText)



        return history, eval_epoch_history

    def runCNN(self, top_words, max_words_number, CNN_Config,userText):
        # embedding_size = 32
        #
        # kernel_size = 3
        # filters = 32
        # pool_size = 2
        #
        # dense_units1 = 250
        # dense_units2 = 1
        #
        # batch_size = 128
        # epochs = 6
        config=CNNConfiguration(CNN_Config[0],CNN_Config[1],CNN_Config[2],CNN_Config[3],
                                CNN_Config[4],CNN_Config[5],CNN_Config[6],CNN_Config[7])
        #config = CNNConfiguration(32, 3, 32, 2, 250, 1, 128, 3)
        cnn=CNN(top_words,max_words_number,config)
        #cnn = CNN(5000, 50, config)
        test_x, test_y, train_x, train_y = loadData(top_words, max_words_number)
        model, history, eval_epoch_history = cnn.defineModel(test_x, test_y, train_x, train_y)

        #model = cnn.loadModel()
        cnn.runModel(model,userText)
        return history, eval_epoch_history

    def runCNN_LSTM(self,top_words,max_words_number, CNN_LSTM_Config,userText):
        # embedding_size = 128
        #
        # # Convolution
        # kernel_size = 5
        # filters = 64
        # pool_size = 4
        #
        # # LSTM
        # lstm_output_size = 70
        #
        # # Training
        # batch_size = 128
        # epochs = 6
        #
        # dropout = 0.25
        # strides = 1
        # dense = 1
        #config = CNN_LSTMConfiguration(128, 5, 64, 4, 70, 128, 3, 0.25, 1, 1)
        config=CNN_LSTMConfiguration(CNN_LSTM_Config[0],CNN_LSTM_Config[1],CNN_LSTM_Config[2],CNN_LSTM_Config[3],CNN_LSTM_Config[4],
                                     CNN_LSTM_Config[5],CNN_LSTM_Config[6],CNN_LSTM_Config[7],CNN_LSTM_Config[8],CNN_LSTM_Config[9])

        #cnn_lstm = CNN_LSTM(5000, 100, config)
        cnn_lstm = CNN_LSTM(top_words, max_words_number,config)

        test_x, test_y, train_x, train_y = loadData(max_words_number, max_words_number)
        model, history, eval_epoch_history = cnn_lstm.defineModel(test_x, test_y, train_x, train_y)

        #model = cnn_lstm.loadModel()
        cnn_lstm.runModel(model,userText)
        return history, eval_epoch_history

    def runBayes(self,top_words,max_words_number,userText):
        training_set, training_labels, validation_set, validation_labels = loadBayesData(top_words,max_words_number)
        print(training_set[0], training_labels[0])
        print(validation_set[0], validation_labels[0])

        NBClassifier = NaiveBayesClassifier()
        model, train_results = NBClassifier.defineModel(validation_set, validation_labels, training_set,
                                                        training_labels)
        NBClassifier.runModel(model,userText)
        return train_results
