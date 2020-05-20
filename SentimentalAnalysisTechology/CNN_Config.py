from ModelConfig import Configuration
class CNNConfiguration(Configuration):
    def __init__(self, *params):
        super().__init__(*params)
        self.embedding_size = params[0]
        self.kernel_size= params[1]
        self.filters= params[2]
        self.pool_size= params[3]
        self.dense_units1= params[4]
        self.dense_units2= params[5]
        self.batch_size= params[6]
        self.epochs= params[7]

    def getConfig(self):
        return self.embedding_size, \
        self.kernel_size, \
        self.filters, \
        self.pool_size, \
        self.dense_units1, \
        self.dense_units2, \
        self.batch_size, \
        self.epochs

    def getConfigAsString(self):
        string_config = super(self.getConfig())
        return string_config