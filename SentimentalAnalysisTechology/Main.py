from LSTM import LSTM
from CNN import CNN
from CNN_LSTM import CNN_LSTM
from NaiveBayesClassifier import NaiveBayesClassifier
from DataLoader import loadData
from DataLoader import loadBayesData
import matplotlib.pyplot as plt
from LSTM_Config import LSTMConfiguration
from CNN_Config import CNNConfiguration
from CNN_LSTM_Config import CNN_LSTMConfiguration


class Main:
    def main(self):
        accList = dict()
        lossList = dict()

        histories = list()

        # bayesResults = self.runBayes()
        # accList.update({"Naive Bayes": bayesResults / 100})
        historyCNN_LSTM, evalCNN_LSTM = self.runCNN_LSTM()

        histories.append(tuple(("CNN_LSTM", historyCNN_LSTM)))
        lossList.update({"CNN_LSTM": evalCNN_LSTM[0]})
        accList.update({"CNN_LSTM": evalCNN_LSTM[1]})
        #
        # historyCNN, evalCNN = self.runCNN()
        #
        # histories.append(tuple(("CNN", historyCNN)))
        # lossList.update({"CNN": evalCNN[0]})
        # accList.update({"CNN": evalCNN[1]})
        #
        # historyLSTM, evalLSTM = self.runLSTM()
        #
        # histories.append(tuple(("LSTM", historyLSTM)))
        # lossList.update({"LSTM": evalLSTM[0]})
        # accList.update({"LSTM": evalLSTM[1]})

        plot = self.outputGraph(histories)
        plot = self.showTestResultsGraph(accList, lossList, plot)

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
        fig.suptitle('Ефективність алгоритмів на текстах з тестової вибірки',y=0.99)
        plot.savefig("graph1.png")
        plot.show()
        return plot

    def outputGraph(self, histories):
        plt.figure( figsize=(13, 5))
        for history in histories:
            algoName, algoHistory = history
            plt.plot(algoHistory.history['loss'], label=('%s training loss' % algoName) )
            plt.plot(algoHistory.history['acc'], label=('%s training accuracy' % algoName))
            plt.legend(loc="upper left")
        plt.xlabel('№ епохи навчання')
        plt.ylabel('Значення')
        plt.title('Ефективність алгоритмів впродовж періоду тренування моделей',y=0.99)
        plt.savefig("graph2.png")
        return plt

    def runLSTM(self):
        config = LSTMConfiguration(32, True, 100, 0.2, 0.2, 1, 128, 3)
        lstm = LSTM(5000, 150, config)
        test_x, test_y, train_x, train_y = loadData(lstm.max_words_number, lstm.max_review_len)

        model, history, eval_epoch_history = lstm.defineModel(test_x, test_y, train_x, train_y)

        # model = lstm.loadModel()

        lstm.runModel(model)
        return history, eval_epoch_history

    def runCNN(self):
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
        config = CNNConfiguration(32, 3, 32, 2, 250, 1, 128, 3)
        cnn = CNN(5000, 150, config)
        test_x, test_y, train_x, train_y = loadData(cnn.max_words_number, cnn.max_review_len)
        model, history, eval_epoch_history = cnn.defineModel(test_x, test_y, train_x, train_y)

        #model = cnn.loadModel()
        cnn.runModel(model)
        return history, eval_epoch_history

    def runCNN_LSTM(self):
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
        config = CNN_LSTMConfiguration(128, 5, 64, 4, 70, 128, 5, 0.25, 1, 1)

        cnn_lstm = CNN_LSTM(5000, 150,config)
        test_x, test_y, train_x, train_y = loadData(cnn_lstm.max_words_number, cnn_lstm.max_review_len)
        model, history, eval_epoch_history = cnn_lstm.defineModel(test_x, test_y, train_x, train_y)

        #model = cnn_lstm.loadModel()
        cnn_lstm.runModel(model)
        return history, eval_epoch_history

    def runBayes(self):
        training_set, training_labels, validation_set, validation_labels = loadBayesData(5000,50)
        print(training_set[0], training_labels[0])
        print(validation_set[0], validation_labels[0])

        NBClassifier = NaiveBayesClassifier(0, 0)
        model, train_results = NBClassifier.defineModel(validation_set, validation_labels, training_set,
                                                        training_labels)
        NBClassifier.runModel(model)
        return train_results

main = Main()
main.main()

