import pathlib

from badass.input.jwst_common import JWSTReader

inst_data_dir = pathlib.Path(__file__).resolve().parent.joinpath('instrument_data', 'nirspec')

FILT_GRAT = {
    '100': '140',
    '170': '235',
    '290': '395',
}


def get_dispersion_file(instrument, io_cfg):
    grating = instrument.grating
    if grating is None:

        filt = instrument.filter
        if filt is None:
            filt = io_cfg.filter
            if filt is None:
                return None

        grating = FILT_GRAT[filt]
        disperser = io_cfg.disperser
        if disperser is None:
            return None

        grating += disperser
        if not grating[0] in ['g','G']:
            grating = 'g' + grating

    inst_data_file = inst_data_dir.joinpath('jwst_nirspec_%s_disp.fits'%(grating.lower()))
    if not inst_data_dir.exists():
        return None
    return inst_data_file



class NIRSpecReader(JWSTReader):

    def set_dispersion(self):
        inst_data_file = get_dispersion_file(self.target.instrument, self.cfg.io)
        if inst_data_file is None:
            # TODO: possible to still do some fitting?
            raise Exception('Could not find dispersion file!')

        self.disp_res = JWSTReader.interp_dispersion(inst_data_file, self.obs_wave, wave_unit=self.wave_unit)

Reader = NIRSpecReader
