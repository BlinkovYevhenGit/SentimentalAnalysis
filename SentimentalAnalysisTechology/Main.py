from LSTM import LSTM
from CNN import CNN
from CNN_LSTM import CNN_LSTM
from NaiveBayesClassifier import NaiveBayesClassifier
from DataLoader import loadData
from DataLoader import loadBayesData
import matplotlib.pyplot as plt


class Main:
    def main(self):
        accList=dict()
        lossList=dict()

        histories = list()

        bayesResults =self.runBayes()
        accList.update({"Naive Bayes":bayesResults/100})
        historyCNN_LSTM, evalCNN_LSTM = self.runCNN_LSTM()

        histories.append(tuple(("CNN_LSTM",historyCNN_LSTM)))
        lossList.update({"CNN_LSTM":evalCNN_LSTM[0]})
        accList.update({"CNN_LSTM":evalCNN_LSTM[1]})

        #
        historyCNN, evalCNN = self.runCNN()

        histories.append(tuple(("CNN",historyCNN)))
        lossList.update({"CNN":evalCNN[0]})
        accList.update({"CNN":evalCNN[1]})
        #
        historyLSTM, evalLSTM = self.runLSTM()

        histories.append(tuple(("LSTM",historyLSTM)))
        lossList.update({"LSTM":evalLSTM[0]})
        accList.update({"LSTM":evalLSTM[1]})
        #
        plot =self.outputGraph(histories)
        plot=self.showTestResultsGraph(accList, lossList,plot)
        plot.show()


    def showTestResultsGraph(self, accList, lossList,plot):
        accNames = list(accList.keys())
        accValues = list(accList.values())
        lossNames = list(lossList.keys())
        lossValues = list(lossList.values())
        plot.figure(2)
        fig, axs = plot.subplots(1, 2, figsize=(9, 3))
        axs[0].bar(accNames, accValues)
        axs[0].set_title("Точність аналізу тексту з тестової вибірки")
        axs[1].bar(lossNames, lossValues)
        axs[1].set_title("Значення функції втрат")
        fig.suptitle('Ефективність алгоритмів аналізу тональності текстів з навчальних вибірок')
        return plot


    def outputGraph(self, histories):
        plt.figure()
        for history in histories:
            algoName, algoHistory = history
            plt.plot(algoHistory.history['loss'], label=('%s training loss' % algoName))
            plt.plot(algoHistory.history['acc'], label=('%s training accuracy' % algoName))
            plt.legend(loc="upper left")
        plt.xlabel('№ епохи навчання')
        plt.ylabel('Значення')
        plt.title('Ефективність алгоритмів впродовж періоду тренування моделей аналізу тональності')
        return plt


    def runLSTM(self):
        lstm = LSTM(5000, 150)
        test_x, test_y, train_x, train_y = loadData(lstm.max_words_number, lstm.max_review_len)

        model,history,eval_epoch_history = lstm.defineModel(test_x, test_y, train_x, train_y)

        #model = lstm.loadModel()

        lstm.runModel(model)
        return history,eval_epoch_history
    def runCNN(self):
        cnn = CNN(5000, 150)
        test_x, test_y, train_x, train_y = loadData(cnn.max_words_number, cnn.max_review_len)
        model,history,eval_epoch_history = cnn.defineModel(test_x, test_y, train_x, train_y)

        #model = cnn.loadModel()
        cnn.runModel(model)
        return history,eval_epoch_history

    def runCNN_LSTM(self):
        cnn_lstm = CNN_LSTM(5000, 150)
        test_x, test_y, train_x, train_y = loadData(cnn_lstm.max_words_number, cnn_lstm.max_review_len)
        model,history,eval_epoch_history = cnn_lstm.defineModel(test_x, test_y, train_x, train_y)

        model = cnn_lstm.loadModel()
        cnn_lstm.runModel(model)
        return history, eval_epoch_history

    def runBayes(self):
        training_set, training_labels, validation_set, validation_labels = loadBayesData()
        NBClassifier = NaiveBayesClassifier(0, 0)
        model,train_results = NBClassifier.defineModel(validation_set, validation_labels,training_set, training_labels)
        NBClassifier.runModel(model)
        return train_results


main1 = Main()
main1.main()
