from LSTM import LSTM
from CNN import CNN
from CNN_LSTM import CNN_LSTM
from NaiveBayesClassifier import NaiveBayesClassifier
from DataLoader import loadData
from DataLoader import loadBayesData


class Main:
    def main(self):
        self.runCNN_LSTM()
        #self.runBayes()
        #self.runLSTM()
        #self.runCNN()


    def runLSTM(self):
        lstm = LSTM(5000, 200)
        test_x, test_y, train_x, train_y = loadData(lstm.max_words_number, lstm.max_review_len)

        # model = lstm.defineModel(test_x, test_y, train_x, train_y)

        model = lstm.loadModel()

        lstm.runModel(model)

    def runCNN(self):
        cnn = CNN(5000, 200)
        # test_x, test_y, train_x, train_y = loadData(cnn.max_words_number, cnn.max_review_len)
        # model = cnn.defineModel(test_x, test_y, train_x, train_y)

        model = cnn.loadModel()
        cnn.runModel(model)

    def runCNN_LSTM(self):
        cnn_lstm = CNN_LSTM(5000, 200)
        # test_x, test_y, train_x, train_y = loadData(cnn_lstm.max_words_number, cnn_lstm.max_review_len)
        # model = cnn_lstm.defineModel(test_x, test_y, train_x, train_y)

        model = cnn_lstm.loadModel()
        cnn_lstm.runModel(model)

    def runBayes(self):
        training_set, training_labels, validation_set, validation_labels = loadBayesData()
        NBClassifier = NaiveBayesClassifier(0, 0)
        model = NBClassifier.defineModel(validation_set, validation_labels,training_set, training_labels)
        NBClassifier.runModel(model)


main1 = Main()
main1.main()
