# importing libraries
import os
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request
from Estimator import Estimator

# creating instance of the class
app = Flask(__name__)


# to tell flask what url shoud trigger the function index()
@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index2.html')
    # return "Hello World"


# prediction function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 12)
    loaded_model = pickle.load(open("model.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]


def getValueFromDictionary(argument, dictionary):
    return dictionary.get(argument, "")


def getConfiguration(userInputConfig, params_number):
    params = []
    params = userInputConfig.split(";", params_number)
    # TODO: validation
    return params[:params_number]


@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        userInputData = request.form.to_dict()
        argument = 0
        LSTM_Input = getValueFromDictionary('LSTM_Input', userInputData)
        combinedInput = getValueFromDictionary('CombinedInput', userInputData)
        CNN_Input = getValueFromDictionary('CNN_Input', userInputData)

        top_words = int(getValueFromDictionary('frequency', userInputData))
        review_max_length = int(getValueFromDictionary('length', userInputData))
        userText=getValueFromDictionary('userInputText',userInputData)
        print(userInputData)
        print(LSTM_Input, combinedInput, CNN_Input, top_words, review_max_length,userText)
        # numbers_to_strings(argument)
        totalConfig = [LSTM_Input, combinedInput, CNN_Input, top_words, review_max_length]

        LSTM_config = getConfiguration(LSTM_Input, 8)
        CNN_config = getConfiguration(CNN_Input, 8)
        Combined_config = getConfiguration(combinedInput, 10)

        print(LSTM_config, CNN_config, Combined_config)
        estimator = Estimator()
        LSTM_config,CNN_config,Combined_config=allconfigsConvertToType(LSTM_config, CNN_config, Combined_config)
        histories,lossList,accList =estimator.estimate(top_words, review_max_length, LSTM_config, CNN_config, Combined_config,userText)
        # userInputData = list(userInputData.values())
        # userInputData = list(map(int, userInputData))
        # print(userInputData)
        # result = ValuePredictor(userInputData)
        #
        # if int(result) == 1:
        #     prediction = 'Income more than 50K'
        # else:
        #     prediction = 'Income less that 50K'

        return render_template("result.html", prediction=[histories[0:],lossList,accList])


def allconfigsConvertToType(lstmConfig, cnn_config, combined_config):
    convertedLSTM=[]
    convertedCNN=[]
    convertedCombined=[]
    try:
        convertedLSTM = [int(lstmConfig[0]), bool(lstmConfig[1]), int(lstmConfig[2]), float(lstmConfig[3]),
                         float(lstmConfig[4]), int(lstmConfig[5]), int(lstmConfig[6]), int(lstmConfig[7])]
        print(convertedLSTM)

        convertedCNN = [int(cnn_config[0]), int(cnn_config[1]), int(cnn_config[2]), int(cnn_config[3]),
                        int(cnn_config[4]), int(cnn_config[5]), int(cnn_config[6]), int(cnn_config[7])]
        print(convertedCNN)

        convertedCombined = [int(combined_config[0]), int(combined_config[1]), int(combined_config[2]), int(combined_config[3]),int(combined_config[4]),
                             int(combined_config[5]), int(combined_config[6]), float(combined_config[7]), int(combined_config[8]), int(combined_config[9])]
        print(convertedCombined)

    except ValueError:
        print("Не вдалося перетворити вхідні дані у необхідний формат")
    return convertedLSTM,convertedCNN,convertedCombined


if __name__ == "__main__":
    app.run(debug=True)
