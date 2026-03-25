from badass.runners.runner import BatchRunner

runners = {}

class IFURunner(BatchRunner):
	area_type = 'general'

runners[IFURunner.area_type] = IFURunner


class SpaxelsRunner(IFURunner):
	area_type = 'spaxels'

runners[SpaxelsRunner.area_type] = SpaxelsRunner


class BinsRunner(IFURunner):
	area_type = 'bins'

runners[BinsRunner.area_type] = BinsRunner


class AperturesRunner(IFURunner):
	area_type = 'apertures'

runners[AperturesRunner.area_type] = AperturesRunner


def get_runner(area_type):
	return runners.get(area_type, None)
