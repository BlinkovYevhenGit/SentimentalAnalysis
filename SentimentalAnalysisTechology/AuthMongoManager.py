from bson import ObjectId
from pymongo import MongoClient
import hashlib

from UsersDataTable import makeUserTable

client = MongoClient("mongodb://localhost:27017/")
db = client['local']
userData = db["Users"]

def saveUser(login,password):
    try:
        password = password.encode()
        print(password)
        passwordEncrypted = hashlib.md5(password).hexdigest()
        configToSave = {"Login": login, "Password": passwordEncrypted, "Access": True}
        userData.insert_one(configToSave)
    except:
        return "Error"
    return "Success"
def loadUser(inputLogin):
    login =""
    password=""
    access=""
    userList=list()
    for record in userData.find({"Login": inputLogin}):
        userList.append(record)
    if len(userList)!=0:
        login=userList[0]["Login"]
        password=userList[0]["Password"]
        access=userList[0]["Access"]
    return login,password,access

def loadAllUsers():
    tableResult=""
    userList = list()
    for record in userData.find():
        userList.append(record)
    if len(userList) == 0:
        return None
    else:
        tableResult=makeUserTable(userList)
    return tableResult

def findRecordById(id):
    x=list()
    for record in userData.find({"_id": ObjectId(id)}):
        x.append(record)
    return x

def updateUserAccess(id,access):
    updateQuery = {"_id": ObjectId(id)}
    newAccess = {"$set": {"Access": access}}
    userData.update_one(updateQuery, newAccess)