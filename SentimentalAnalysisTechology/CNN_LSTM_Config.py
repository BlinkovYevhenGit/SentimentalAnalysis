from ModelConfig import Configuration
class CNN_LSTMConfiguration(Configuration):
    def __init__(self, *params):
        super().__init__(*params)
        self.embedding_size = params[0]
        self.kernel_size = params[1]
        self.filters = params[2]
        self.pool_size = params[3]
        self.lstm_output_size = params[4]
        self.batch_size = params[5]
        self.epochs = params[6]
        self.dropout = params[7]
        self.strides = params[8]
        self.dense = params[9]

    def getConfig(self):
        return  self.embedding_size, \
        self.kernel_size, \
        self.filters, \
        self.pool_size, \
        self.lstm_output_size, \
        self.batch_size, \
        self.epochs,\
        self.dropout,\
        self.strides,\
        self.dense

    def getConfigAsString(self):
        string_config = super(self.getConfig())
        return string_config