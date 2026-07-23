from __future__ import annotations
import astropy.units as u
from dataclasses import dataclass
from importlib import import_module
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import prodict
from typing import NamedTuple

from spark.io.models import Coord, SparkSpec, SparkCube, SparkSpaxel, SparkCircularAperture, SparkRectangularAperture
from spark.utils import redden, deredden

from badass.utils.config import BadassConfig
import badass.utils.constants as constants
from badass.utils.logger import BadassLogger
from badass.utils.pca import pca_reconstruction
from badass.utils.utils import ccm_unred, get_ebv, emline_masker, log_rebin, metal_masker

# TODO: use a dataclass to explicitly define expected attrs and make sure all input classes have consistent attrs
# TODO: set up pre-input creation logger

TARGET_WAVE_UNIT = u.AA
TARGET_FLUX_UNIT_AA = u.erg / u.s / (u.cm**2) / u.AA


class FitReg(NamedTuple):
    min: float
    max: float

    def __str__(self):
        return f'({self.min}, {self.max})'


    def __repr__(self):
        return self.__str__()


@dataclass
class BadassSpec(SparkSpec):
    name: str = None
    cfg: BadassConfig = None
    obs_wave: np.ndarray = None
    fit_reg: FitReg = None
    flux_norm: float = None
    disp_res: int | float | np.ndarray = None
    velscale: float = None
    valid: bool = True
    err_log: str = ''

    def __post_init__(self):
        print('BadassSpec __post_init__')
        super().__post_init__()

        if self.name is None:
            if not self.cfg.io.product_name is None:
                self.name = self.cfg.io.product_name
            elif not self.file is None:
                self.name = self.file.stem
            else:
                self.name = 'spec-%d'%int(time.time() * 1000)

        for attr,unit in {'wave':TARGET_WAVE_UNIT, 'obs_wave':TARGET_WAVE_UNIT, 'flux':TARGET_FLUX_UNIT_AA, 'err':TARGET_FLUX_UNIT_AA}.items():
            attr_val = getattr(self, attr)
            if isinstance(attr_val, u.Quantity):
                setattr(self, attr, attr_val.to(unit).value)
        self.wave_unit = TARGET_WAVE_UNIT
        self.flux_unit = TARGET_FLUX_UNIT_AA

        if self.wave_is_rest:
            self.obs_wave = redden(self.wave, z=self.target.z)
        else:
            self.obs_wave = self.wave.copy()
            self.wave = deredden(self.obs_wave, z=self.target.z)

        if isinstance(self.disp_res, (float,int)):
            self.disp_res = np.full(len(self.wave), self.disp_res)


    def postinit(self):

        self.set_fit_region()
        if self.fit_reg is None:
            self.valid = False
            return

        reg_mask = ((self.wave >= self.fit_reg.min) & (self.wave <= self.fit_reg.max))
        self.flux = self.flux[reg_mask]
        self.wave = self.wave[reg_mask]
        self.obs_wave = self.obs_wave[reg_mask]
        self.err = self.err[reg_mask]
        self.disp_res = self.disp_res[reg_mask]

        nan_flux = np.where(~np.isfinite(self.flux))[0]
        nan_err = np.where(~np.isfinite(self.err))[0]
        inan = np.unique(np.concatenate([nan_flux,nan_err]))
        # Interpolate over nans and infs if in galaxy or err
        self.err[inan] = np.nan
        self.err[inan] = 1.0 if all(np.isnan(self.err)) else np.nanmedian(self.err)

        fit_mask_bad = []
        if self.cfg.fit.mask_bad_pix:
            self.bad_pix = getattr(self, 'bad_pix', np.array([]))
            fit_mask_bad.extend(self.bad_pix)
        if self.cfg.fit.mask_emline:
            fit_mask_bad.extend(emline_masker(self.wave,self.flux,self.err))
        for m in self.cfg.user_mask:
            fit_mask_bad.extend(np.where((self.wave >= m[0]) & (self.wave <= m[1]))[0])
        if self.cfg.fit.mask_metal:
            fit_mask_bad.extend(metal_masker(self.wave,self.flux,self.err))

        ebv = get_ebv(self.target.ra, self.target.dec)
        self.flux = ccm_unred(self.obs_wave, self.flux, ebv)

        self.fit_norm = np.nanmax(self.flux)
        self.flux = self.flux / self.fit_norm
        self.err = self.err / self.fit_norm
        self.err[self.err == 0] = np.nanmedian(self.err)

        if self.cfg.get('pca', {}).get('do_pca',False):
            pca_reconstruction(self) # TODO: test

        if np.isnan(self.flux).all():
            self.valid = False
            self.err_log = '\'flux\' array is all nans, not running fit'
            return

        fit_mask_bad.extend(np.where(np.isnan(self.flux))[0])
        fit_mask_bad.extend(np.where(np.isnan(self.err))[0])
        fit_mask_bad = np.sort(np.unique(fit_mask_bad))
        self.fit_mask = np.setdiff1d(np.arange(0,len(self.wave),1,dtype=int),fit_mask_bad)

        if self.cfg.io.dust_cache != None:
            IrsaDust.cache_location = str(dust_cache)


    def set_fit_region(self):
        # Determines the fitting region for an input spectrum and fit options
        # Fitting region initially the edges of wavelength vector
        self.fit_reg = FitReg(min=self.wave[0], max=self.wave[-1])
        self.log.info('Initial fitting region: {fr}'.format(fr=self.fit_reg))

        user_fit_reg = self.cfg.fit.fit_reg
        if isinstance(user_fit_reg, (tuple,list)):
            user_fit_reg = FitReg(*user_fit_reg)
            if user_fit_reg.min > user_fit_reg.max:
                self.log.error('Fitting boundaries overlap!')
                self.fit_reg = None
                return

            if (user_fit_reg.min > self.fit_reg.max) or (user_fit_reg.max < self.fit_reg.min):
                self.log.error('Fitting region not available!')
                self.fit_reg = None
                return

            if (user_fit_reg.min < self.fit_reg.min) or (user_fit_reg.max > self.fit_reg.max):
                self.log.warn('Input fitting region exceeds available wavelength range. BADASS will adjust your fitting range automatically...')
                self.log.warn('\t- Input fitting range: %s'%str(user_fit_reg))
                self.log.warn('\t- Available wavelength range: %s'%str(self.fit_reg))

            self.fit_reg = FitReg(np.max([user_fit_reg.min, self.fit_reg.min]), np.min([user_fit_reg.max, self.fit_reg.max]))
        elif (isinstance(user_fit_reg, str)) and (user_fit_reg == 'auto'):
            self.log.info('Auto setting fitting region')
        else:
            self.log.error('Invalid fitting region')
            self.fit_reg = None
            return

        # The lower limit of the spectrum must be the lower limit of our stellar templates
        # TODO: template function to let each template affect the fitting region?
        if self.cfg.comp.fit_losvd:
            min_losvd = constants.LOSVD_LIBRARIES[self.cfg.losvd.library].min_losvd
            max_losvd = constants.LOSVD_LIBRARIES[self.cfg.losvd.library].max_losvd
            if (self.fit_reg.min < min_losvd) or (self.fit_reg.max > max_losvd):
                self.log.warn('Warning: Fitting LOSVD requires wavelenth range between {mi} Å and {ma} Å for stellar templates. BADASS will adjust your fitting range to fit the LOSVD...'.format(mi=min_losvd, ma=max_losvd))
                self.log.warn('\t- Available wavelength range: ',(self.fit_reg))
            self.fit_reg = FitReg(np.max([min_losvd, self.fit_reg.min]), np.min([max_losvd, self.fit_reg.max]))

        self.log.info('- New fitting region is {fr}'.format(fr=self.fit_reg))
        if (self.fit_reg.max - self.fit_reg.min) < constants.MIN_FIT_REGION:
            self.log.error('Fitting region too small! The fitting region must be at least {min_reg} A!'.format(min_reg=constants.MIN_FIT_REGION))
            self.fit_reg = None
            return


    @classmethod
    def from_dict(cls, input_data, cfg=prodict.Prodict({})):
        # if (len(cfg) == 0) and (not input_data.get('cfg', None) is None):
        #     cfg = prodict.Prodict(input_data['cfg'])
        return cls.from_format(input_data, cfg)


    @classmethod
    def parse(cls, input_data, cfg):
        print('BadassSpec parse')
        return cls.from_fits(input_data, cfg=cfg, z=cfg.fit.redshift)


    @classmethod
    def from_format(cls, input_data, cfg):
        fmt = cfg.io.infmt+'_reader'

        try:
            module = import_module('badass.input.'+fmt)
        except ImportError as e:
            raise Exception('Could not find Reader Module: %s (%s)' % (fmt,e))

        if not getattr(module, 'Reader', None):
            raise Exception('No Reader specified in %s' % fmt)

        return module.Reader.parse(input_data, cfg)


    @classmethod
    def from_path(cls, _path, cfg, filter=None):
        # TODO: implement support to filter different types
        #       of files from the supplied directory

        path = pathlib.Path(_path)
        if not path.exists():
            raise Exception('Unable to find input path: %s' % str(path))

        if path.is_file():
            return cls.from_format(path, cfg)

        inputs = []
        # TODO: add search string option and recursion option
        for infile in path.glob('*'):
            # TODO: support recursion into subdirs?
            if not infile.is_file():
                continue

            ret = cls.from_format(infile, cfg)
            inputs.extend(ret if isinstance(ret, list) else [ret])
        return inputs


    @classmethod
    def get_inputs(cls, input_data, cfg):
        # TODO: from_previous_run

        if isinstance(input_data, cls):
            if not cfg is None:
                input_data.cfg = cfg
            return input_data

        if isinstance(input_data, SparkSpec):
            return cls.from_spark(input_data, cfg)

        if isinstance(input_data, list):

            if isinstance(cfg, list) and (len(cfg) != 1 and len(cfg) != len(input_data)):
                raise Exception('Options list must be same length as input data')

            if isinstance(cfg, BadassConfig):
                cfg = [cfg] * len(input_data)
            elif len(cfg) == 1:
                cfg = [cfg[0]] * len(input_data)

            inputs = []
            for ind, opt in zip(input_data, cfg):
                res = cls.get_inputs(ind, opt)
                if isinstance(res, list):
                    inputs.extend(res)
                else:
                    inputs.append(res)
            return inputs

        if isinstance(input_data, dict):
            return cls.from_dict(input_data, cfg)

        if isinstance(input_data, pathlib.Path):
            return cls.from_path(input_data, cfg)

        # Check if string path
        if isinstance(input_data, str):
            if pathlib.Path(input_data).exists():
                return cls.from_path(input_data, cfg)

        return cls.from_format(input_data, cfg)


    def set_new_logger(self):
        self.log = BadassLogger(self)


@dataclass
class BadassSpaxel(BadassSpec, SparkSpaxel):
    pass


@dataclass
class BadassCircularAperture(BadassSpec, SparkCircularAperture):
    pass


@dataclass
class BadassRectangularAperture(BadassSpec, SparkRectangularAperture):
    pass


@dataclass
class BadassCube(BadassSpec, SparkCube):
    spaxel_class = BadassSpaxel

    ap_shapes = {
        'circular': BadassCircularAperture,
        'rectangular': BadassRectangularAperture,
    }


# TODO: there's probably a better way than making these...
@dataclass
class LogRebinMixin(BadassSpec):

    def log_rebin(self):
        lam_range = (np.min(self.wave),np.max(self.wave))
        self.flux, log_lam, self.velscale = log_rebin(self.wave, self.flux, velscale=None, flux=False, oversample=self.cfg.fit.log_rebin_oversample)
        self.err, _, _ = log_rebin(self.wave, self.err, velscale=self.velscale, flux=False, oversample=self.cfg.fit.log_rebin_oversample)
        self.wave = np.exp(log_lam)
        self.obs_wave = redden(self.wave, z=self.target.z)


@dataclass
class LogRebinSpaxel(BadassSpaxel, LogRebinMixin):
    pass


@dataclass
class LogRebinCircularAperture(BadassCircularAperture, LogRebinMixin):
    pass


@dataclass
class LogRebinRectangularAperture(BadassRectangularAperture, LogRebinMixin):
    pass

