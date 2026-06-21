import pathlib

from badass.input.jwst_common import JWSTReader

inst_data_dir = pathlib.Path(__file__).resolve().parent.joinpath('instrument_data', 'nirspec')

FILT_GRAT = {
    '100': '140',
    '170': '235',
    '290': '395',
}

class NIRSpecReader(JWSTReader):

    def set_dispersion(self):
        grating = FILT_GRAT[self.cfg.io.filter]
        disperser = self.cfg.io.disperser.lower()
        inst_data_file = inst_data_dir.joinpath('jwst_nirspec_g%s%s_disp.fits'%(grating,disperser))
        self.disp_res = JWSTReader.interp_dispersion(inst_data_file, self.obs_wave, wave_unit=self.wave_unit)

Reader = NIRSpecReader
