import pathlib

from badass.input.jwst_common import JWSTReader

inst_data_file = pathlib.Path(__file__).resolve().parent.joinpath('instrument_data', 'miri', 'MIRI_INST_DATA_FULL.fits')

class MIRIReader(JWSTReader):

    def set_dispersion(self):
        self.disp_res = JWSTReader.interp_dispersion(inst_data_file, self.obs_wave, wave_unit=self.wave_unit)

Reader = MIRIReader
