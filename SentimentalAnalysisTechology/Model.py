
class Model:
    def __init__(self,*params):
        self.configuration=params
        pass

    def toJSON(self):
        pass

    def defineModel(self, test_x, test_y, train_x, train_y):
        pass

    def runModel(self, model,userText):
        pass

    def loadModel(self):
        pass

    def saveModel(self, model):
        pass

# TODO: add loading data for all algorithms

# TODO: add saving and loading model

# TODO: Somehow add configuration of NNs

# TODO: somehow create db connection

# TODO: consider what to store in db

# TODO: consider how to view structure of input/output data because .npz and .h5 are too complicated
