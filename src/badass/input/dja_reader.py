from dataclasses import dataclass
import numpy as np

from spark.io.readers.jwst import DJASpec
from spark.utils import deredden

from badass.input.input import LogRebinMixin
from badass.input.jwst_common import JWSTReader
from badass.input.nirspec_reader import inst_data_dir, FILT_GRAT

@dataclass
class DJAReader(LogRebinMixin, DJASpec):

    def __post_init__(self):
        super().__post_init__()

        div = int(np.floor(np.log10(np.abs(np.nanmedian(self.flux)))))
        self.flux_norm = 10**div
        self.flux = self.flux / self.flux_norm
        self.err = self.err / self.flux_norm
        self.log_rebin()

        if self.cfg.io.filter == 'clear' or self.cfg.io.grating == 'prism':
            inst_data_file = inst_data_dir.joinpath('jwst_nirspec_prism_disp.fits')
        else:
            grating = self.cfg.io.grating
            if grating is None:
                grating = FILT_GRAT[self.cfg.io.filter]
            disperser = self.cfg.io.disperser.lower()
            inst_data_file = inst_data_dir.joinpath('jwst_nirspec_g%s%s_disp.fits'%(grating,disperser))
        self.disp_res = JWSTReader.interp_dispersion(inst_data_file, self.obs_wave, wave_unit=self.wave_unit)
        self.disp_res = deredden(self.disp_res, self.target.z)


Reader = DJAReader
