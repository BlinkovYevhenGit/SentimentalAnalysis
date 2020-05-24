# importing libraries
import os
from flask import Flask, render_template, request, redirect, flash
import hashlib
import AuthMongoManager
from AuthMongoManager import updateUserAccess, loadAllUsers, saveUser, loadUser
from Estimator import Estimator, deleteSavedImages
from MongoManager import loadConfiguration, findRecordById, makeModelTable, makeResultTable, makeReportTable

app = Flask(__name__)

graphFolder = os.path.join('static', 'graphs')

# use decorators to link the function to a url
@app.route('/')
def home():
    return render_template('login.html')  # return a string

# @app.route('/welcome')
# def welcome():
#     return render_template('welcome.html')  # render a template

# Route for handling the login page logic
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None

    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['password'] != 'admin':
            if request.form['username'].strip() == '' or request.form['password'].strip() == '':
                error = 'Одне з текстових полів пусте!'
            else:
                encryptedPassword, login, access = findUserData(request.form['username'])
                if request.form['username'] != login:
                    error = 'Користувача з даним логіном не існує'
                else:
                    input = request.form['password'].encode()
                    print(input)
                    inputEncrypted = hashlib.md5(input).hexdigest()
                    print(inputEncrypted)
                    if inputEncrypted != encryptedPassword:
                        error = 'Пароль невірний!'
                    else:
                        if access == True:
                            deleteSavedImages()
                            return render_template('AnalysisPage.html')
                        else:
                            error = "У доступі до моделей відмовлено!"
        else:
            userTable=loadAllUsers()
            return render_template("AdminPage.html",userTable=userTable)
    return render_template('login.html', error=error)


def findUserData(login):
    foundLogin, password, access = loadUser(login)
    return password, foundLogin, access


# start the server with the 'run()' method

@app.route('/register', methods=['GET','POST'])
def register():
    error = None
    if request.method =='GET':
        return render_template('register.html')
    else:
        if request.form['password']!=request.form['repeatPassword']:
            error = 'Введені паролі не збігаються!'
            return render_template('register.html',error=error)
        foundLogin,password,access=loadUser(request.form['username'])

        if(len(foundLogin)!=0):
            if request.form['username']==foundLogin:
                error='Користувач з таким логіном вже існує!'
                return render_template('register.html', error=error)

        if request.form['username'].strip() == '' or request.form['password'].strip() == '' or request.form['repeatPassword'].strip() == '':
            error='Одне з текстових полів пусте!'
            return render_template('register.html', error=error)

        saveUser(request.form['username'], request.form['password'])
        deleteSavedImages()
        return render_template('AnalysisPage.html')

@app.route('/item/<string:id>', methods=['GET', 'POST'])
def change(id):
    record = AuthMongoManager.findRecordById(id)
    userAccess= not bool(record[0]["Access"])
    updateUserAccess(id, userAccess)
    userTable =loadAllUsers()
    return render_template("AdminPage.html",userTable=userTable)
# creating instance of the class

dataTable = dict()
chosenModels = dict()
LSTM_Table = []
CNN_LSTM_Table = []
CNN_Table = []
text = ""


# to tell flask what url shoud trigger the function index()
@app.route('/AnalysisPage')
def index():
    return render_template('AnalysisPage.html')
    # return "Hello World"


@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


def getValueFromDictionary(argument, dictionary):
    value = None
    try:
        value = dictionary.get(argument, "")
    except:
        pass
    return value


def getConfiguration(userInputConfig, params_number):
    params = []
    params = userInputConfig.split(";", params_number)
    # TODO: validation
    return params[:params_number]


@app.route('/load', methods=['POST'])
def loadData():
    global text
    userInputData = request.form.to_dict()
    text = getValueFromDictionary('inputText', userInputData)
    global LSTM_Table
    global CNN_LSTM_Table
    global CNN_Table
    lstmInput, LSTM_Table = loadConfiguration("lstm_model")
    dataTable.update({"LSTM": LSTM_Table})
    cnnInput, CNN_Table = loadConfiguration("cnn_model")
    dataTable.update({"CNN": CNN_Table})
    cnn_lstmInput, CNN_LSTM_Table = loadConfiguration("cnn_lstm_model")
    dataTable.update({"CNN_LSTM": CNN_LSTM_Table})
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
    LSTM_Table.border = True
    return render_template('dataTable.html', LSTM_table=LSTM_Table, CNN_table=CNN_Table, CNN_LSTM_table=CNN_LSTM_Table)


@app.route('/cell/<string:id>', methods=['GET', 'POST'])
def choose(id):
    global LSTM_Table
    global CNN_LSTM_Table
    global CNN_Table
    record = findRecordById(id)
    recordTable = makeModelTable(record)
    modelType = record[0]['ModelName']
    if modelType == "lstm_model":
        LSTM_Table = recordTable
        chosenModels.update({'LSTM': record[0]})
    if modelType == "cnn_model":
        CNN_Table = recordTable
        chosenModels.update({'CNN': record[0]})
    if modelType == "cnn_lstm_model":
        CNN_LSTM_Table = recordTable
        chosenModels.update({'CNN_LSTM': record[0]})
    print(record)
    return render_template('dataTable.html', LSTM_table=LSTM_Table, CNN_table=CNN_Table, CNN_LSTM_table=CNN_LSTM_Table)

@app.route('/run', methods=['POST'])
def runModels():
    estimator = Estimator()
    filenames = []
    review_len_s = []
    filenames.append(getValueFromDictionary('ModelFileName', getValueFromDictionary('LSTM', chosenModels)))
    filenames.append(getValueFromDictionary('ModelFileName', getValueFromDictionary('CNN', chosenModels)))
    filenames.append(getValueFromDictionary('ModelFileName', getValueFromDictionary('CNN_LSTM', chosenModels)))
    review_len_s.append(getValueFromDictionary('MaxReviewLen', getValueFromDictionary('LSTM', chosenModels)))
    review_len_s.append(getValueFromDictionary('MaxReviewLen', getValueFromDictionary('CNN', chosenModels)))
    review_len_s.append(getValueFromDictionary('MaxReviewLen', getValueFromDictionary('CNN_LSTM', chosenModels)))
    results, tableResult = estimator.runAll(filenames, review_len_s, text)
    print(results)
    table = makeResultTable(tableResult)
    return render_template("result.html", text=text, table=table)

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        userInputData = request.form.to_dict()
        Bayes_Input = getValueFromDictionary('Bayes_Input', userInputData)
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
        histories, lossList, accList, predictions, text_results, tableResult,report = estimator.estimate(top_words,
                                                                                                  review_max_length,
                                                                                                  Bayes_Input,
                                                                                                  LSTM_config,
                                                                                                  CNN_config,
                                                                                                  Combined_config,
                                                                                                  userText)
        # userInputData = list(userInputData.values())
        # userInputData = list(map(int, userInputData))
        # print(userInputData)
        # result = ValuePredictor(userInputData)
        #
        # if int(result) == 1:
        #     prediction = 'Income more than 50K'
        # else:
        #     prediction = 'Income less that 50K'
        table = makeResultTable(tableResult)
        reportTable=makeReportTable(report)
        graph1_filename = os.path.join(graphFolder, 'graph1.png')
        graph2_filename = os.path.join(graphFolder, 'graph2.png')

        return render_template("result.html", text=userText,
                               # baes=predictions[0], cnn_lstm=predictions[1],
                               # cnn=predictions[2], lstm=predictions[3],
                               graph1=graph1_filename,
                               graph2=graph2_filename, table=table, reportTable=reportTable)  # [histories[0:],lossList,accList]


def allconfigsConvertToType(lstmConfig, cnn_config, combined_config):
    convertedLSTM = []
    convertedCNN = []
    convertedCombined = []
    if len(lstmConfig) != 1:
        try:
            convertedLSTM = [int(lstmConfig[0]), bool(lstmConfig[1]), int(lstmConfig[2]), float(lstmConfig[3]),
                             float(lstmConfig[4]), int(lstmConfig[5]), int(lstmConfig[6]), int(lstmConfig[7])]
            print(convertedLSTM)
        except ValueError:
            flash("Не вдалося перетворити конфігурацію ДКЧП у необхідний формат")
            return redirect("/AnalysisPage")
    if len(cnn_config) != 1:
        try:
            convertedCNN = [int(cnn_config[0]), int(cnn_config[1]), int(cnn_config[2]), int(cnn_config[3]),
                            int(cnn_config[4]), int(cnn_config[5]), int(cnn_config[6]), int(cnn_config[7])]
            print(convertedCNN)
        except ValueError:
            flash("Не вдалося перетворити конфігурацію Комбінованої нейронної мережі у необхідний формат")
            return redirect("/AnalysisPage")
    if len(combined_config) != 1:
        try:

            convertedCombined = [int(combined_config[0]), int(combined_config[1]), int(combined_config[2]),
                                 int(combined_config[3]), int(combined_config[4]),
                                 int(combined_config[5]), int(combined_config[6]), float(combined_config[7]),
                                 int(combined_config[8]), int(combined_config[9])]
            print(convertedCombined)
        except ValueError:
            flash("Не вдалося перетворити конфігурацію Згорткової нейронної мережі у необхідний формат")
            return redirect("/AnalysisPage")
    return convertedLSTM, convertedCNN, convertedCombined

if __name__ == "__main__":
    app.run(debug=True)
