class IFUPipeline(BadassPipeline):
    area_type = 'general'


# TODO: different spaxels/bins/apertures can have different configs
class SpaxelsPipeline(IFUPipeline):
    area_type = 'spaxels'

    def __init__(self, target, cfg):
        super().__init__(target, cfg)

        self.spaxel_results = {}


    def run(self):

        

        for spaxel in spaxels:
            BadassPipeline().run()

        gather_data()


class BinsPipeline(IFUPipeline):
    area_type = 'bins'


class AperturesPipeline(IFUPipeline):
    area_type = 'apertures'


def get_ifu_type(area_type):
    for pipeline in [IFUPipeline, SpaxelsPipeline, BinsPipeline, AperturesPipeline]:
        if pipeline.area_type == area_type:
            return pipeline
    return None