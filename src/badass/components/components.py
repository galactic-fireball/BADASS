
class BadassComponent:

    def __init__(self, ctx):
        self.ctx = ctx
        self.pr = self.ctx.param_reg
        self.br = self.ctx.blob_reg
        self.comp_params = []
        self.initialize_parameters()
        self.register_blobs()


    @classmethod
    def initialize_component(cls):
        return None


    def initialize_parameters(self):
        pass


    def register_blobs(self):
        pass


    def add_components(self, comp_dict, host_model):
        return host_model


    def get_param(self, param_name):
        return np.nan
