# importing libraries
import os

import flask
from flask import Flask, render_template, request, flash,redirect
from Estimator import Estimator, deleteSavedImages
from MongoManager import loadConfiguration, findRecordById, makeModelTable

graphFolder = os.path.join('static', 'graphs')

# creating instance of the class
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = graphFolder



dataTable = dict()
chosenModels = dict()
LSTM_Table = []
CNN_LSTM_Table = []
CNN_Table=[]

# to tell flask what url shoud trigger the function index()
@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index2.html')
    # return "Hello World"


@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


def getValueFromDictionary(argument, dictionary):
    return dictionary.get(argument, "")


def getConfiguration(userInputConfig, params_number):
    params = []
    params = userInputConfig.split(";", params_number)
    # TODO: validation
    return params[:params_number]


@app.route('/load', methods=['GET'])
def loadData():
    global LSTM_Table
    global CNN_LSTM_Table
    global CNN_Table
    lstmInput,LSTM_Table = loadConfiguration("lstm_model")
    dataTable.update({"LSTM":LSTM_Table})
    cnnInput,CNN_Table = loadConfiguration("cnn_model")
    dataTable.update({"CNN":CNN_Table})
    cnn_lstmInput, CNN_LSTM_Table = loadConfiguration("cnn_lstm_model")
    dataTable.update({"CNN_LSTM":CNN_LSTM_Table})
    # review_lenLSTM = lstmInput[0]['MaxReviewLen']
    # #review_lenCNN =  cnnInput[0]['MaxReviewLen']
    # review_lenCNN_LSTM = cnn_lstmInput[0]['MaxReviewLen']
    # # if review_lenLSTM != review_lenCNN & review_lenLSTM != review_lenCNN_LSTM:
    # #     flash("Довжина відгуку у конфігураціях всіх моделей неоднакова!")
    # #     return redirect('/')
    # top_wordsLSTM=lstmInput[0]['TopWords']
    # #top_wordsCNN=cnnInput[0]['TopWords']
    # top_wordsCNN_LSTM=cnn_lstmInput[0]['TopWords']
    #
    # lstmConfig=lstmInput[0]['Configuration']
    # #cnnConfig=cnnInput[0]['Configuration']
    # cnn_lstmConfig=cnn_lstmInput[0]['Configuration']
    #
    # result = [lstmConfig, "", cnn_lstmConfig]
    # print(lstmConfig)
    # #print(cnnConfig)
    # print(cnn_lstmConfig)
    LSTM_Table.border=True
    return render_template('dataTable.html',LSTM_table=LSTM_Table,CNN_Table = CNN_Table,CNN_LSTM_table=CNN_LSTM_Table)

@app.route('/item/<string:id>', methods=['GET', 'POST'])
def choose(id):
    global LSTM_Table
    global CNN_LSTM_Table
    global CNN_Table
    record=findRecordById(id)
    recordTable = makeModelTable(record)
    modelType=record[0]['ModelName']
    if modelType=="lstm_model":
        LSTM_Table=recordTable
        chosenModels.update({'LSTM':record[0]})
    if modelType=="cnn_model":
        CNN_Table=recordTable
        chosenModels.update({'CNN':record[0]})
    if modelType=="cnn_lstm_model":
        CNN_LSTM_Table=recordTable
        chosenModels.update({'CNN_LSTM':record[0]})
    print(record)
    return render_template('dataTable.html', LSTM_table=LSTM_Table,CNN_Table=CNN_Table,CNN_LSTM_table=CNN_LSTM_Table)


@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        userInputData = request.form.to_dict()
        LSTM_Input = getValueFromDictionary('LSTM_Input', userInputData)
        combinedInput = getValueFromDictionary('CombinedInput', userInputData)
        CNN_Input = getValueFromDictionary('CNN_Input', userInputData)

        top_words = int(getValueFromDictionary('frequency', userInputData))
        review_max_length = int(getValueFromDictionary('length', userInputData))
        userText = getValueFromDictionary('inputText', userInputData)
        print(userInputData)
        print(LSTM_Input, combinedInput, CNN_Input, top_words, review_max_length, userText)
        # numbers_to_strings(argument)
        totalConfig = [LSTM_Input, combinedInput, CNN_Input, top_words, review_max_length]

        LSTM_config = getConfiguration(LSTM_Input, 8)
        CNN_config = getConfiguration(CNN_Input, 8)
        Combined_config = getConfiguration(combinedInput, 10)

        print(LSTM_config, CNN_config, Combined_config)
        estimator = Estimator()
        LSTM_config, CNN_config, Combined_config = allconfigsConvertToType(LSTM_config, CNN_config, Combined_config)
        histories, lossList, accList, predictions = estimator.estimate(top_words, review_max_length, LSTM_config,
                                                                       CNN_config, Combined_config, userText)
        # userInputData = list(userInputData.values())
        # userInputData = list(map(int, userInputData))
        # print(userInputData)
        # result = ValuePredictor(userInputData)
        #
        # if int(result) == 1:
        #     prediction = 'Income more than 50K'
        # else:
        #     prediction = 'Income less that 50K'
        graph1_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'graph1.png')
        graph2_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'graph2.png')

        return render_template("result.html", text=userText, baes=predictions[0], cnn_lstm=predictions[1],
                               cnn=predictions[2], lstm=predictions[3], graph1=graph1_filename,
                               graph2=graph2_filename)  # [histories[0:],lossList,accList]


def allconfigsConvertToType(lstmConfig, cnn_config, combined_config):
    convertedLSTM = []
    convertedCNN = []
    convertedCombined = []
    try:
        convertedLSTM = [int(lstmConfig[0]), bool(lstmConfig[1]), int(lstmConfig[2]), float(lstmConfig[3]),
                         float(lstmConfig[4]), int(lstmConfig[5]), int(lstmConfig[6]), int(lstmConfig[7])]
        print(convertedLSTM)

        convertedCNN = [int(cnn_config[0]), int(cnn_config[1]), int(cnn_config[2]), int(cnn_config[3]),
                        int(cnn_config[4]), int(cnn_config[5]), int(cnn_config[6]), int(cnn_config[7])]
        print(convertedCNN)

        convertedCombined = [int(combined_config[0]), int(combined_config[1]), int(combined_config[2]),
                             int(combined_config[3]), int(combined_config[4]),
                             int(combined_config[5]), int(combined_config[6]), float(combined_config[7]),
                             int(combined_config[8]), int(combined_config[9])]
        print(convertedCombined)

    except ValueError:
        print("Не вдалося перетворити вхідні дані у необхідний формат")
    return convertedLSTM, convertedCNN, convertedCombined


if __name__ == "__main__":
    deleteSavedImages()
    app.run(debug=True)
