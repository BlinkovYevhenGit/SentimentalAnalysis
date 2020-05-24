from ModelConfig import Configuration
class LSTMConfiguration(Configuration):
    def __init__(self, *params):
        super().__init__(*params)
        self.embed_vec_len = params[0]
        self.mask_zero = params[1]
        self.units = params[2]
        self.dropout = params[3]
        self.recurrent_dropout = params[4]
        self.dense_units = params[5]
        self.bat_size = params[6]
        self.max_epochs = params[7]

    def getConfig(self):
        return self.embed_vec_len, \
               self.mask_zero, \
               self.units,\
               self.dropout,\
               self.recurrent_dropout,\
               self.dense_units,\
               self.bat_size,\
               self.max_epochs

    def getConfigAsString(self):
        string_config = super(self.getConfig())
        return string_config