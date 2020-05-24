import gridfs
from bson import ObjectId
from pymongo import MongoClient
import os

from tensorflow import keras as K

from DataTable import DataTable, ModelTable, ResultTable, ReportTable

client = MongoClient("mongodb://localhost:27017/")
db = client['local']
modelsConfig = db["Configuration"]
fs = gridfs.GridFS(db)


def deleteAllModels():
    files = db["fs.files"]
    chunks = db["fs.chunks"]
    x = files.delete_many({})
    y = chunks.delete_many({})
    print(x.deleted_count, " - documents deleted", y.deleted_count, " - chunks deleted")


def saveToDB(filename, model):
    print("Saving model to database \n")
    mp = ".\\Models\\" + filename
    model.save(mp)
    fileID = fs.put(open((r'Models\%s' % filename).replace('\\', '/'), 'rb'), filename=filename)
    os.remove(mp)
    return fileID


def loadModelFromDB(filepath):
    out = fs.find_one({"_id": ObjectId(str(filepath))})
    weights_temp = out.read()
    f_out_path = (r'Models\\' + out.name).replace('\\', '/')
    with open(f_out_path, 'wb') as f:
        f.write(weights_temp)
    model = K.models.load_model(".\\Models\\%s" % out.name)
    return model


def saveConfiguration(modelName, top_words, review_len, model_config, filename):
    configToSave = {"ModelName": modelName, "TopWords": top_words, "MaxReviewLen": review_len,
                    "Configuration": model_config, "ModelFileName": filename}
    fileid = modelsConfig.insert_one(configToSave)
    return fileid


def loadConfiguration(modelName):
    x=list()

    for record in modelsConfig.find({'ModelName': modelName},{ "_id": 1,'ModelName':1, 'TopWords':1, 'MaxReviewLen':1,'Configuration':1, 'ModelFileName':1}):
        x.append(record)
    # print(x)
    # r=dict()
    # r.update({'ModelName': x['ModelName'], 'TopWords': x['TopWords']})
    # print(r)
    # items = [dict(name='Name1', description='Description1'),
    #         dict(name='Name2', description='Description2'),
    #         dict(name='Name3', description='Description3')]
    # items2=[{'Name':'Name1','Description':'Description1','Phone':'Phone1'},{'Name':'Name2','Description':'Description2','Phone':'phone2'},{'Name':'Name3','Description':'Description3','Phone':'phone3'}]

    table=DataTable(x)
    return x,table
def findRecordById(id):
    x=list()
    for record in modelsConfig.find({"_id": ObjectId(id)}):
        x.append(record)
    return x

def makeModelTable(data):
    return ModelTable(data)

def makeResultTable(data):
    return ResultTable(data)
def makeReportTable(data):
    return ReportTable(data)

