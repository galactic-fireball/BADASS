from dataclasses import dataclass
import numpy as np

from spark.io.readers.jwst import DJASpec
from spark.utils import deredden

from badass.input.input import LogRebinMixin, BadassSpec
from badass.input.jwst_common import JWSTReader
from badass.input.nirspec_reader import inst_data_dir, get_dispersion_file

@dataclass
class DJAReader(BadassSpec, DJASpec, LogRebinMixin):

    def __post_init__(self):
        super().__post_init__()

        div = int(np.floor(np.log10(np.abs(np.nanmedian(self.flux)))))
        self.flux_norm = 10**div
        self.flux = self.flux / self.flux_norm
        self.err = self.err / self.flux_norm
        self.log_rebin()

        inst_data_file = get_dispersion_file(self.target.instrument, self.cfg.io)
        if inst_data_file is None:
            # TODO: possible to still do some fitting?
            raise Exception('Could not find dispersion file!')

        self.disp_res = JWSTReader.interp_dispersion(inst_data_file, self.obs_wave, wave_unit=self.wave_unit)
        self.disp_res = deredden(self.disp_res, self.target.z)


Reader = DJAReader
