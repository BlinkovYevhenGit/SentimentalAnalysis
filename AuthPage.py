# import the Flask class from the flask module
from flask import Flask, render_template, request, url_for,redirect,blueprints
import hashlib
# create the application object
from AuthMongoManager import findRecordById, updateUserAccess, loadAllUsers, saveUser, loadUser
from Estimator import deleteSavedImages

app = Flask(__name__)

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
    record = findRecordById(id)
    userAccess= not bool(record[0]["Access"])
    updateUserAccess(id, userAccess)
    userTable =loadAllUsers()
    return render_template("AdminPage.html",userTable=userTable)



if __name__ == '__main__':
    app.run(debug=True)