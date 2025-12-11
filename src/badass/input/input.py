from importlib import import_module
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import prodict
import time
import shutil

import badass.utils.constants as constants
from badass.utils.logger import BadassLogger
from badass.utils.pca import pca_reconstruction
from badass.utils.utils import ccm_unred, emline_masker, get_ebv, metal_masker

# TODO: use a dataclass to explicitly define expected attrs and make sure all input classes have consistent attrs
# TODO: set up pre-input creation logger

class BadassInput():

    def __init__(self, input_data, options):
        # TODO: make dataclass
        if not hasattr(self, 'valid'):
            self.valid = True
        if not hasattr(self, 'err_log'):
            self.err_log = ''

        if (not hasattr(self, 'wave')) or (not hasattr(self, 'spec')):
            self.valid = False
            self.err_log = 'Can\'t fit spec without \'wave\' or \'spec\' values'
            print(self.err_log)
            return

        if (not hasattr(self, 'ra')) or (not hasattr(self, 'dec')):
            print('WARNING: ra and/or dec not set, the galactic average E(B-V) will used')
            self.ra, self.dec = None, None

        if not hasattr(self, 'flux_norm'):
            self.flux_norm = 1.0

        if not hasattr(self, 'options'):
            self.options = options

        if not hasattr(self, 'name'):
            if hasattr(self.options.io_options, 'product_name'):
                self.name = self.options.io_options.product_name
            elif hasattr(self, 'infile'):
                self.name = self.infile.stem
            else:
                self.name = 'spec-%d'%int(time.time() * 1000)

        if not hasattr(self, 'outdir'):
            if hasattr(self.options.io_options, 'output_dir'):
                self.outdir = pathlib.Path(self.options.io_options.output_dir)
                if self.options.io_options.get('multi', False):
                    self.outdir = self.outdir.joinpath(self.name)
            elif hasattr(self, 'infile'):
                self.outdir = self.infile.parent.resolve().joinpath(self.name)
            else:
                self.outdir = pathlib.Path(os.getcwd()).resolve().joinpath(self.name)
        if not self.outdir.is_absolute():
            self.outdir = pathlib.Path(os.getcwd()).resolve().joinpath(self.outdir)

        # TODO: check for fit completed
        if self.outdir.exists():
            if self.options.io_options.get('overwrite', False):
                print('Removing old output directory: [%s]'%str(self.outdir))
                shutil.rmtree(str(self.outdir))
            else:
                self.valid = False
                self.err_log = 'Output directory [%s] already exists, not overwriting'%str(self.outdir)
                print(self.err_log)
                return


    def postinit(self):

        self.outdir.mkdir(parents=True, exist_ok=True)
        self.outdir.joinpath('log').mkdir(parents=True, exist_ok=True) # TODO: 'log' mkdir eventually happens in separate output class

        self.set_fit_region()
        if self.fit_reg is None:
            self.valid = False
            return

        if isinstance(self.disp_res, (float,int)):
            self.disp_res = np.full(len(self.wave), self.disp_res)

        self.bad_pix = getattr(self, 'bad_pix', np.array([]))
        reg_mask = ((self.wave >= self.fit_reg.min) & (self.wave <= self.fit_reg.max))
        self.spec = self.spec[reg_mask]
        self.wave = self.wave[reg_mask]
        self.noise = self.noise[reg_mask]
        self.disp_res = self.disp_res[reg_mask]

        nan_gal = np.where(~np.isfinite(self.spec))[0]
        nan_noise = np.where(~np.isfinite(self.noise))[0]
        inan = np.unique(np.concatenate([nan_gal,nan_noise]))
        # Interpolate over nans and infs if in galaxy or noise
        self.noise[inan] = np.nan
        self.noise[inan] = 1.0 if all(np.isnan(self.noise)) else np.nanmedian(self.noise)

        fit_mask_bad = []
        if self.options.fit_options.mask_bad_pix:
            fit_mask_bad.extend(self.bad_pix)
        if self.options.fit_options.mask_emline:
            fit_mask_bad.extend(emline_masker(self.wave,self.spec,self.noise))
        for m in self.options.user_mask:
            fit_mask_bad.extend(np.where((self.wave >= m[0]) & (self.wave <= m[1]))[0])
        if self.options.fit_options.mask_metal:
            fit_mask_bad.extend(metal_masker(self.wave,self.spec,self.noise))

        ebv = get_ebv(self.ra, self.dec)
        self.spec = ccm_unred(self.wave, self.spec, ebv)

        self.fit_norm = np.round(np.nanmax(self.spec), 5)
        self.spec = self.spec / self.fit_norm
        self.noise = self.noise / self.fit_norm
        self.noise[self.noise == 0] = np.nanmedian(self.noise)

        if self.options.pca_options.do_pca:
            pca_reconstruction(self) # TODO: test

        if np.isnan(self.spec).all():
            self.valid = False
            self.err_log = '\'spec\' array is all nans, not running fit'
            return

        fit_mask_bad.extend(np.where(np.isnan(self.spec))[0])
        fit_mask_bad.extend(np.where(np.isnan(self.noise))[0])
        fit_mask_bad = np.sort(np.unique(fit_mask_bad))
        self.fit_mask = np.setdiff1d(np.arange(0,len(self.wave),1,dtype=int),fit_mask_bad)


    def set_fit_region(self):
        # Determines the fitting region for an input spectrum and fit options

        # Fitting region initially the edges of wavelength vector
        self.fit_reg = (self.wave[0], self.wave[-1])
        self.log.info('Initial fitting region: ({mi}, {ma})'.format(mi=self.fit_reg[0], ma=self.fit_reg[1]))

        # TODO: set default to 'auto' in options schema
        user_fit_reg = self.options.fit_options.fit_reg
        if isinstance(user_fit_reg, (tuple,list)):
            if user_fit_reg[0] > user_fit_reg[1]:
                self.log.error('Fitting boundaries overlap!')
                self.fit_reg = None
                return

            if (user_fit_reg[0] > self.fit_reg[1]) or (user_fit_reg[1] < self.fit_reg[0]):
                self.log.error('Fitting region not available!')
                self.fit_reg = None
                return

            if (user_fit_reg[0] < self.fit_reg[0]) or (user_fit_reg[1] > self.fit_reg[1]):
                self.log.warn('Input fitting region exceeds available wavelength range. BADASS will adjust your fitting range automatically...')
                self.log.warn('\t- Input fitting range: (%d, %d)' % (user_fit_reg[0], user_fit_reg[1]))
                self.log.warn('\t- Available wavelength range: (%d, %d)' % (self.fit_reg[0], self.fit_reg[1]))

            self.fit_reg = (np.max([user_fit_reg[0], self.fit_reg[0]]), np.min([user_fit_reg[1], self.fit_reg[1]]))
        elif (isinstance(user_fit_reg, str)) and (user_fit_reg == 'auto'):
            self.log.info('Auto setting fitting region')
        else:
            self.log.error('Invalid fitting region')
            self.fit_reg = None
            return


        # The lower limit of the spectrum must be the lower limit of our stellar templates
        # TODO: template function to let each template affect the fitting region?
        if self.options.comp_options.fit_losvd:
            min_losvd = constants.LOSVD_LIBRARIES[self.options.losvd_options.library].min_losvd
            max_losvd = constants.LOSVD_LIBRARIES[self.options.losvd_options.library].max_losvd
            if (self.fit_reg[0] < min_losvd) or (self.fit_reg[1] > max_losvd):
                self.log.warn("Warning: Fitting LOSVD requires wavelenth range between {mi} Å and {ma} Å for stellar templates. BADASS will adjust your fitting range to fit the LOSVD...".format(mi=min_losvd, ma=max_losvd))
                self.log.warn("\t- Available wavelength range: (%d, %d)" % (self.fit_reg[0], self.fit_reg[1]))
            self.fit_reg = (np.max([min_losvd, self.fit_reg[0]]), np.min([max_losvd, self.fit_reg[1]]))

        # allow for more explicit variable name: fit_reg.min and fit_reg.max
        # self.fit_reg = type('FitReg', (object,), dict(min=self.fit_reg[0], max=self.fit_reg[1]))
        self.fit_reg = prodict.Prodict({'min':self.fit_reg[0],'max':self.fit_reg[1]})
        self.log.info("- New fitting region is ({mi}, {ma})".format(mi=self.fit_reg.min, ma=self.fit_reg.max))

        if (self.fit_reg.max - self.fit_reg.min) < constants.MIN_FIT_REGION:
            self.log.error('Fitting region too small! The fitting region must be at least {min_reg} A!'.format(min_reg=constants.MIN_FIT_REGION))
            self.fit_reg = None
            return

        mask = ((self.wave >= self.fit_reg.min) & (self.wave <= self.fit_reg.max))
        igood = np.where((self.spec[mask]>0) & (self.noise[mask]>0))[0]
        good_frac = (len(igood)*1.0)/len(self.spec[mask])
        if good_frac < self.options.fit_options.good_thresh:
            self.log.error('Not enough good channels above threshold!')
            self.fit_reg = None
            return


    @classmethod
    def from_dict(cls, input_data, options=prodict.Prodict({})):
        if (len(options) == 0) and (not input_data.get('options', None) is None):
            options = prodict.Prodict(input_data['options'])

        if options.get('io_options', None) is None:
            options.io_options = {}
        if options.io_options.get('infmt', None) is None:
            options.io_options.infmt = 'default'
        return cls.from_format(input_data, options)


    @classmethod
    def parse(cls, input_data, options):
        return cls(input_data, options)


    @classmethod
    def from_format(cls, input_data, options):
        options = options if isinstance(options, dict) else options[0]
        fmt = options.io_options.infmt+'_reader'

        try:
            module = import_module('badass.input.'+fmt)
        except ImportError as e:
            raise Exception('Could not find Reader Module: %s (%s)' % (fmt,e))

        if not getattr(module, 'Reader', None):
            raise Exception('No Reader specified in %s' % fmt)

        readers = module.Reader.parse(input_data, options)
        readers = readers if isinstance(readers, list) else [readers]
        # valid_readers = []
        # for reader in readers:
        #     if not reader.valid:
        #         continue
        #     reader.postinit()
        #     if reader.valid:
        #         valid_readers.append(reader)
        #     # TODO: log invalid readers
        # return valid_readers

        return readers


    @classmethod
    def from_path(cls, _path, options, filter=None):
        # TODO: implement support to filter different types
        #       of files from the supplied directory

        path = pathlib.Path(_path)
        if not path.exists():
            raise Exception('Unable to find input path: %s' % str(path))

        if path.is_file():
            return cls.from_format(path, options)

        inputs = []
        # TODO: add search string option and recursion option
        for infile in path.glob('*'):
            # TODO: support recursion into subdirs?
            if not infile.is_file():
                continue

            ret = cls.from_format(infile, options)
            inputs.extend(ret if isinstance(ret, list) else [ret])
        return inputs


    @classmethod
    def get_inputs(cls, input_data, options):
        # TODO: from_previous_run
        if isinstance(input_data, list):

            if isinstance(options, list) and (len(options) != 1 and len(options) != len(input_data)):
                raise Exception('Options list must be same length as input data')

            opts = options
            if isinstance(options, dict):
                opts = [options] * len(input_data)
            elif len(options) == 1:
                opts = [options[0]] * len(input_data)

            inputs = []
            for ind, opt in zip(input_data, opts):
                opt.io_options.multi = True
                inputs.extend(cls.get_inputs(ind, opt))
            return inputs

        if isinstance(input_data, dict):
            ret = cls.from_dict(input_data, options)
            return ret if isinstance(ret, list) else [ret]

        if isinstance(input_data, pathlib.Path):
            ret = cls.from_path(input_data, options)
            return ret if isinstance(ret, list) else [ret]

        # Check if string path
        if isinstance(input_data, str):
            if pathlib.Path(input_data).exists():
                ret = cls.from_path(input_data, options)
                return ret if isinstance(ret, list) else [ret]
            # if not, could be actual data

        ret = cls.from_format(input_data, options)
        return ret if isinstance(ret, list) else [ret]


    def set_new_logger(self):
        self.log = BadassLogger(self)


    def validate_input(self):
        # Custom input parsers or input dict should provide these values
        # TODO: further validation for each value?
        # TODO: check fit_reg
        # TODO: need infile?
        for attr in ['infile', 'ra', 'dec', 'z', 'wave', 'spec', 'noise', 'disp_res']:
            if not hasattr(self, attr) or getattr(self, attr) is None:
                raise Exception('BADASS input missing expected value: {attr}'.format(attr=attr))

        return True


# TODO: use default_reader.py?
# TODO: make Mixin instead so targets can still have their associated class
class CustomReader(BadassInput):
    def __init__(self, input_data, options):
        self.__dict__.update(input_data)
        if not hasattr(self, 'options'):
            self.options = options
        super().__init__(input_data, options)

