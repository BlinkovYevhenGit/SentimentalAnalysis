
class Configuration:
    def __init__(self, *params):
        self.config = params

    def getConfig(self):
        return self.config
#All the same:
    #embedding_size
    #bat_size
    # #max_epochs

#CNN_LSTM and LSTM
    #dropout
    #dense
    #units

#CNN_LSTM and CNN
    #kernel_size
    #filters
    #pool_size
    #dense_units

#LSTM and CNN
    #dence_units