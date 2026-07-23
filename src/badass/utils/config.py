import importlib
import importlib.util
import json
import pathlib
import prodict
from pydantic import AfterValidator, AliasChoices, BeforeValidator, BaseModel, ConfigDict, DirectoryPath, Field, FiniteFloat, model_validator, NonNegativeInt, NonNegativeFloat, PositiveInt, PositiveFloat, TypeAdapter
from typing import Annotated, Any, ClassVar, List, Literal, types, Union

import badass.utils.constants as consts
from badass.components.spectral_lines.line_profiles import line_profiles

PositiveNum = PositiveInt | PositiveFloat
NonNegativeNum = NonNegativeInt | NonNegativeFloat


# TODO: add as after validator when necessary
def is_lohi(v: list | tuple) -> list:
    if (len(v) == 2) and (v[0]<=v[1]): return v
    raise ValueError('Value is not lo hi')


class CustomBaseModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    is_component: ClassVar[bool] = False
    alt_doc_name: ClassVar[str] = None
    description: ClassVar[str] = None

    @model_validator(mode='after')
    def inject_param_defaults(self):
        for name, value in self.dict().items():
            if not isinstance(value, dict):
                continue

            field_default = self.model_fields[name].default
            if not isinstance(field_default, dict):
                continue

            for k,v in field_default.items():
                if not k in value:
                    value[k] = v
            self[name] = value

        return self


    def __getitem__(self, key:str) -> Any:
        return getattr(self, key)


    def __setitem__(self, key:str, val:Any) -> None:
        object.__setattr__(self, key, val)


    def get(self, key:str, default=None) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        return default


def validate_infmt(v:str) -> str:
    if v in consts.SUPPORTED_INSTRUMENTS:
        return v
    raise ValueError('infmt \'%s\' not in %s'%(v,consts.SUPPORTED_INSTRUMENTS))


def to_path(v:Any) -> pathlib.Path:
    if v is None:
        return v

    try:
        path = pathlib.Path(v)
        if not path.is_absolute():
            path = pathlib.Path.cwd().joinpath(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    except:
        raise ValueError('Could not convert %s to Path'%v)


class IOOptions(CustomBaseModel):
    infmt: Literal[*(consts.SUPPORTED_INSTRUMENTS+['default',])] = Field(None, json_schema_extra={'required':True}, description='The format of the input file. Currently supported options: `[\'sdss\', \'muse\', \'nirspec\', \'miri\']`')
    output_dir: Annotated[DirectoryPath, BeforeValidator(to_path)] = Field(None, description='The output directory of the BADASS results, logs, plots, etc.')
    product_name: str = Field(None, description='')
    overwrite: bool = Field(False, description='If `True`, overwrite the `output_dir` if it already exists.')
    nprocesses: int = Field(1, description='For runs of multiple spectra or IFU cubes, run in multiprocess mode.')
    out_fmt: Literal['fits', 'json', 'csv'] = Field('fits', description='The output format.')
    verbose_out: bool = Field(False, description='Return full available output.')
    log_level: str = Field('info', description='The output log level. Options: `[\'debug\', \'info\', \'warning\', \'error\', \'critical\']`')
    filter: str = Field(None, description='The filter of the provided NIRSpec data cube.')
    grating: str = Field(None, description='The grating of the provided NIRSpec data cube.')
    disperser: str = Field(None, description='The disperser of the provided NIRSpec data cube.')
    dust_cache: Annotated[DirectoryPath | None, BeforeValidator(to_path)] = Field(None, description='Directory path to cache of Irsa dust extinction data.')


def validate_fitreg(v:list|str) -> list:
    if isinstance(v, str):
        if v != 'auto':
            raise ValueError('fit_reg \'%s\' not allowed. Must be range or \'auto\'.')
        return [0,100000]
    return is_lohi(v)


def dict_to_prodict(v:dict|prodict.Prodict) -> prodict.Prodict:
    return prodict.Prodict(v)


class Cosmology(CustomBaseModel):
    H0: float = 70.0
    Om0: float = 0.30


# 'bins': {'side_length':3, 'x': (10,40), 'y':(10,35), 'method': 'mean'},
class BinSpec(CustomBaseModel):
    side_length: int = 0
    method: str = 'sum'
    x: list | tuple = (0,-1)
    y: list | tuple = (0,-1)


# 'apertures': {'shape':'rectangular', 'center': (10,40), 'width':12,},
class ApSpec(CustomBaseModel):
    shape: str
    center: tuple | list
    method: str = 'sum'
    width: float = -1.0
    height: float = -1.0
    radius: float = -1.0


class FitArea(CustomBaseModel):
    # TODO: after validator to change spaxel to spaxels for consistency
    type: str = None
    spaxels: dict | list | str = Field(default=None, alias=AliasChoices('spaxels','spaxel'))
    bins: BinSpec = Field(default=BinSpec(), alias=AliasChoices('bins','bin'))
    apertures: ApSpec | list[ApSpec] = Field(default=None, alias=AliasChoices('apertures','aperture'))
    plot_input: bool = False


class FitOptions(CustomBaseModel):
    fit_reg: Annotated[list[NonNegativeNum] | str, AfterValidator(validate_fitreg)] = Field('auto', description='The minimum and maximum desired fitting wavelength in angstroms.', examples=[(4400,5500), '\'auto\''])
    redshift: NonNegativeNum = Field(0.0, description='Redshift of the fitting target')
    log_rebin_oversample: NonNegativeNum = Field(1, description='')
    skip_bootstrap: bool = Field(False, description='Option to skip max likelihood bootstrapping')
    n_basinhop: NonNegativeInt = Field(25, description='Number of successive `niter_success` times the basinhopping algorithm needs to achieve a solution. The fit becomes much better with more success times, however this can increase the time to a solution significantly.')
    max_like_niter: NonNegativeInt = Field(10, description='Number of bootstrapping iterations to perform after the initial basinhopping fit. This is a means to obtain uncertainties on parameters without performing MCMC fitting, however, do not produce as robust uncertainties as MCMC.')
    # TODO: fit_area specific definition
    fit_area: FitArea = Field(default=FitArea(), description='Defines the area to be fit for data cubes. See `examples/muse_examples.py` for usage.')
    reweighting: bool = Field(False, description='If `True`, BADASS will reweight the noise vector to achieve a reduced chi-squared ~ 1. This is done after the initial basinhopping fit, and applied to any bootstrapped uncertainties and MCMC fitting performed afterward. This does not affect the chi-squared ratio metric used in line and configuration testing, but does effect the amplitude-over-noise and SNR calculations in BADASS.')
    fit_stat: str = Field('ML', description='The fit statistic used for the likelihood. Options:\n\n* `\'ML\'` for standard maximum likelihood (pixels weighted by noise with no noise scaling).\n* `\'OLS\'` for ordinary least-squares fitting (all pixels weighted by same amount).')
    cosmology: Cosmology = Field(default=Cosmology(), description='The flat Lambda-CDM cosmology assumed for calculating luminosities from fluxes.')
    test_models: bool = Field(False, description='Performs tests for lines. Options are specified in `test_options`.')
    mask_emline: bool = Field(False, description='Mask any significant absorption and emission features relative to the continuum. This uses an automated iterative moving median filter of various sizes to detect significant flux differences between window sizes. Good for continuum fitting but tends to over mask lots of features near the edges of the spectrum.')
    mask_bad_pix: bool = Field(False, description='Mask pixels which the specified instrument has flagged as bad due to sky line subtraction or cosmic rays.')
    mask_metal: bool = Field(False, description='Performs the same moving median filter algorithm as `mask_emline` but only to absorption features. Works well for metal absorption features seen typically in high-redshift spectra.')
    feature_edge_pad: NonNegativeNum = 10


class MCMCOptions(CustomBaseModel):
    mcmc_fit: bool = Field(False, description='Perform fit with MCMC using the initial maximum likelihood fit as initial parameters for the fit. It is *highly recommended* that one use MCMC to perform the fit, although sampling will require a significant amount of time compared to a maximum likelihood fit using `scipy.optimize.minimize()`.')
    nwalkers: NonNegativeInt = Field(100, description='Number of "walkers" per parameter used by emcee to explore each parameter space. The minimum number of walkers is 2 x ( # of free parameters), set by emcee.')
    auto_stop: bool = Field(True, description='If `True`, autocorrelation is used to automatically stop the fitting process when a convergence criteria (`conv_type`) is achieved. ')
    conv_type: str | list[str] = Field('median', description='Mode of convergence. Convergence of \'all\' ensures all fitting parameters have achieved the desired `ncor_times` and `autocorr_tol` criteria, while "median" and "mean" only ensure that `ncor_times` and `autocorr_tol` criteria have been met for the median or mean of all parameters, respectively. A list of valid parameters is also acceptable to ensure specific parameters have achieved convergence even if others have not. In general "median" requires the fewest number of iterations and is not sensitive to poorly-constrained parameters, and "all" and "mean" require the most number of iterations and are much more sensitive to fluctuations in calculated autocorrelation times and tolerances. A list of parameters is suitable in cases where one is only interested in certain spectral features.', examples=['\'all\'', '\'median\'', '\'mean\'', ['NA_OIII_5007_AMP', 'NA_OIII_5007_DISP',]])
    min_samp: NonNegativeInt = Field(1000, description='If `auto_stop=True`, then the `burn_in` is the iteration at which convergence is achieved, and `min_samp` is the number of iterations *past convergence* used for posterior sampling (the samples used for histograms and estimating best-fit parameters and uncertainties). If for some reason the parameters "jump out" of convergence, the `burn_in` will reset and BADASS will continue to sample until convergence is met again. If emcee completes `min_samp` iterations after convergence is achieved without jumping out of convergence, this concludes the MCMC sampling.')
    ncor_times: NonNegativeInt = Field(10, description='The number of integrated autocorrelation times (iterations) needed for convergence. We recommend a minimum of `ncor_times=2.0`. In general, it will require more than 2.0 autocorrelation times to calculate the autocorrelation time for a parameter chain. Increasing `ncor_times` ensures that the parameter chain has stopped exploring the parameter space and is ready to begin sampling for the posterior distribution.')
    autocorr_tol: NonNegativeNum = Field(10, description='The percent change in the current integrated autocorrelation time and the previously calculated integrated autocorrelation time. The `write_iter` determines how often BADASS checks a parameter\'s integrated autocorrelation time. In general, we find that `autocorr_tol=5` (a 5%% change) is acceptable for a converged parameter chain. A parameter chain that diverges more than 10%% in 100 iterations could still be exploring the parameter space for a stable solution. A `autocorr_tol=1` (a 1%% change) typically requires many more iterations than necessary for convergence.')
    write_iter: NonNegativeInt = Field(100, description='The frequency at which BADASS writes the current parameter values (median walker positions). If `auto_stop=True`, then BADASS checks for convergence every `write_iter` iteration for convergence.')
    write_thresh: NonNegativeInt = Field(100, description='The iteration at which writing (and checking for convergence if `auto_stop=True`) begins. BADASS does not check for convergence before this value.')
    burn_in: NonNegativeInt = Field(1500, description='If `auto_stop=False` then this serves as the burn-in for a maximum number of iterations. If `auto_stop=True`, this value is ignored.')
    min_iter: NonNegativeInt = Field(100, description='The minimum number of iterations BADASS performs before it is allowed to stop. This is true regardless of the value of `auto_stop`.')
    max_iter: NonNegativeInt = Field(2500, description='The maximum number of iterations BADASS performs before stopping. This value is adhered to regardless of the value of `auto_stop` to set a limit on the number of iterations before BADASS should "give up."')


class CompOptions(CustomBaseModel):
    is_component = True
    alt_doc_name = 'components'

    fit_feii: bool = Field(True, description='Broad and narrow optical FeII templates are taken from [Véron-Cetty et al. (2004)](https://ui.adsabs.harvard.edu/abs/2004A%26A...417..515V/abstract) with each line modeled using a Gaussian. One can also optionally using the template from [Kovačević et al. (2010)](https://ui.adsabs.harvard.edu/abs/2010MSAIS..15..176K/abstract), however with limited coverage (4400 Å - 5500 Å). FeII emission can be very strong in some Type 1 (broad line) AGN, but is almost undetectable in Type 2 (narrow line) AGN.')
    fit_uv_iron: bool = Field(False, description='Fits the empirical UV iron template from [Vestergaard and Wilkes (2001)](https://ui.adsabs.harvard.edu/abs/2001ApJS..134....1V/abstract), for high-redshift spectra with coverage < 3500 Å.')
    fit_balmer: bool = Field(False, description='Fits a series of higher-order Balmer lines and Balmer pseudo-continuum for high-redshift spectra with coverage < 3500 Å.')
    fit_losvd: bool = Field(True, description='Stellar line-of-sight velocity distribution (LOSVD) using Penalized Pixel-Fitting ([pPXF](https://www-astro.physics.ox.ac.uk/~mxc/software/#ppxf), [Cappellari et al. (2017)](https://ui.adsabs.harvard.edu/abs/2017MNRAS.466..798C/abstract)) using templates from the [Indo-U.S. Library of Coudé Feed Stellar Spectra](https://www.noao.edu/cflib/) ([Valdes et al. (2004)](https://ui.adsabs.harvard.edu/abs/2004ApJS..152..251V/abstract)) in the optical region 3460 Å - 9464 Å. This is used to obtain stellar kinematics in spectra with resolvable absorption features, such as stellar velocity and dispersion.')
    fit_host: bool = Field(False, description='Fits a host galaxy template using single-stellar population templates from the EMILES library. Note that this method does not estimate stellar LOSVD, but can shift in velocity and convolve to match the data as best as it can.')
    fit_power: bool = Field(True, description='This fits a power-law component to simulate the effect of the AGN "blue-bump" continuum.')
    fit_poly: bool = Field(False, description='Fit a polynomial continuum component of a specified order. Polynomial options are specified by `poly_options` dictionary. Options are additive Legendre polynomial or multiplicative Legendre polynomial. The order must be within the range 0 <= order <= 99. Note: caution should be used when using polynomial components, as these can be degenerate with other continuum components, and higher-order polynomials can lead to overfitting.')
    fit_narrow: bool = Field(True, description='Fit lines of the `line_type: \'na\'` in the line list. Narrow forbidden emission lines are seen in both Type 1 and Type 2 AGNs, as well as starforming galaxies.')
    fit_broad: bool = Field(True, description='Fit lines of the `line_type: \'br\'` in the line list. Broad permitted emission lines are commonly seen in Type 1 AGN.')
    fit_absorp: bool = Field(False, description='Fit lines of the `line_type: \'abs\'` in the line list. Occasionally one might need to fit a strong absorption feature that isn\'t described by stellar processes, such as a broad absorption line in a quasar.')
    tie_line_disp: bool = Field(False, description='Ties the widths of all respective line types (all narrow lines are tied, all broad lines are tied, etc.). This can be done to significantly reduce the number of free parameters in the fit if fitting many lines, however it is not recommended.')
    tie_line_voff: bool = Field(False, description='Ties the velocity offsets of all respective line types (all narrow lines are tied, all broad lines are tied, etc.). This can be done to significantly reduce the number of free parameters in the fit if fitting many lines, however it is not recommended.')

    def fit(self, comp_name):
        fit_name = 'fit_'+comp_name
        if not hasattr(self, fit_name):
            return False
        return getattr(self, fit_name)


    def tie(self, attr):
        tie_name = 'tie_line_'+attr
        if not hasattr(self, tie_name):
            return False
        return getattr(self, tie_name)


Number = int | FiniteFloat
NonNegativeParam = dict[str, list[str | NonNegativeNum] | str | NonNegativeNum | dict] | str | NonNegativeNum
Param = dict[str, list[str | Number] | str | Number | dict] | str | Number

# TODO: make amp after validator to normalize if > 1

class PowerOptions(CustomBaseModel):
    is_component = True

    type: Literal['simple', 'broken'] = Field('simple', examples=['\'simple\'', '\'broken\''])
    amp: Param = {'init':'0.5*median_flux', 'plim':(0, 'max_flux')}
    slope: Param = {'init':-1.0, 'plim':(-6.0, 6.0)}
    break_: Param = Field({'init':'max_wave - 0.5*(max_wave-min_wave)', 'plim':('min_wave', 'max_wave')}, alias='break') # alias to handle Python reserved word
    slope_1: Param = {'init': -1.0, 'plim':(-6.0, 6.0)}
    slope_2: Param = {'init': -1.0, 'plim':(-6.0, 6.0)}
    curvature: Param = {'init': 0.1, 'plim':(0.01, 1.0)}


class PolyOptions(CustomBaseModel):
    is_component = True
    alt_doc_name = 'polynomial'

    apoly_order: int = 0
    apoly_coeff: Param | list[Number] = {'init': 0.0, 'plim':(-1.0e2, 1.0e2)}
    mpoly_order: int = 0
    mpoly_coeff: Param | list[Number] = {'init': 0.0, 'plim':(-1.0e2, 1.0e2)}


class LOSVDOptions(CustomBaseModel):
    is_component = True

    library: Literal[*list(consts.LOSVD_LIBRARIES.keys())] = 'IndoUS'
    vel: Param = {'init':100.0, 'plim':(-500.0, 500.0)}
    disp: NonNegativeParam = {'init':150.0, 'plim':(0.001, 500.0)}



class HostOptions(CustomBaseModel):
    is_component = True
    description = '''
        The host model is used as a simplified placeholder in the event that the stellar 
        continuum isn't of any interest. These are single-stellar population templates from 
        the EMILES library, and do not have a low enough resolution for reliable stellar LOSVD fitting.
    '''

    age: list[PositiveNum] = [0.1,1.0,10.0]
    amp: NonNegativeParam = {'init':'0.5*median_flux', 'plim':(0.0,'max_flux')}
    vel: Param = {'init':0.0, 'plim':(-500.0, 500.0)}
    disp: NonNegativeParam = {'init':100.0, 'plim':(0.001, 500.0)}


class OptFeIIOptions(CustomBaseModel):
    is_component = True
    alt_doc_name = 'optical_feii'
    description = '''
        There are two FeII templates built into BADASS. The default is the broad and narrow 
        templates from [Véron-Cetty et al. (2004)](https://ui.adsabs.harvard.edu/abs/2004A%26A...417..515V/abstract) (`VC04`). 
        This model allows the user to have amplitude, dispersion, and velocity offset as free-parameters, 
        with options to constrain them to constant values during the fit.  BADASS can also use the temperature-dependent 
        template from [Kovačević et al. (2010)](https://ui.adsabs.harvard.edu/abs/2010MSAIS..15..176K/abstract) (`K10`), 
        which allows for the fitting of individual F, S, G, and I Zw 1 atomic transitions, as well as temperature. 
        The K10 template is best suited for modeling FeII in NLS1 objects with strong FeII emission.
    '''

    template: Literal['VC04', 'K10'] = 'VC04'


class VC04_Options(OptFeIIOptions):
    template: Literal['VC04'] = 'VC04'
    na_amp: NonNegativeParam = {'init':'0.1*median_flux', 'plim':(0, 'max_flux')}
    br_amp: NonNegativeParam = {'init':'0.1*median_flux', 'plim':(0, 'max_flux')}
    na_disp: NonNegativeParam = {'init':10.0, 'plim':(0.1, 250.0)}
    br_disp: NonNegativeParam = {'init':500.0, 'plim':(100, 5000.0)}
    na_voff: Param = {'init':0.0, 'plim':(-1000.0, 1000.0)}
    br_voff: Param = {'init':0.0, 'plim':(-2000.0, 2000.0)}


class K10_Options(OptFeIIOptions):
    template: Literal['K10'] = 'K10'
    f_amp: NonNegativeParam = {'init':'0.001*median_flux', 'plim':(0, '0.1*max_flux')}
    s_amp: NonNegativeParam = {'init':'0.001*median_flux', 'plim':(0, '0.1*max_flux')}
    g_amp: NonNegativeParam = {'init':'0.001*median_flux', 'plim':(0, '0.1*max_flux')}
    z_amp: NonNegativeParam = {'init':'0.001*median_flux', 'plim':(0, '0.1*max_flux')}
    disp: NonNegativeParam = {'init': 250.0, 'plim':(0.1, 2500.0)}
    voff: Param = {'init': 0.0, 'plim':(-1000.0, 1000.0)}
    temp: NonNegativeParam = {'init': 10000.0, 'plim':(2000.0, 20000.0)}


class PlotOptions(CustomBaseModel):
    html: bool = True
    param_hist: bool = True
    corner: bool = True


class OutputOptions(CustomBaseModel):
    write_chain: bool = True


class SpectralLine(CustomBaseModel):
    name: str = ''
    type: Literal['narrow', 'broad', 'absorb', 'combined'] = 'narrow'
    center: float | int | None = None


class BaseLine(SpectralLine):

    # General hyperpars; child classes can override
    amp: NonNegativeParam = {'init':'0.001*median_flux', 'plim':(0.0,1.0)}
    amp_adjust: bool = True # allow BADASS to adjust amp depending on surrounding features
    disp: NonNegativeParam = {'init':50.0, 'plim':(0.001,500.0)}
    voff: Param = {'init':0.0, 'plim':(-500.0,500.0), 'prior':{'type':'gaussian'}}
    voff_adjust: bool = True # allow BADASS to adjust voff depending on surrounding features

    # Higher-order moments for Gauss-Hermite, Laplace, and Uniform line profiles
    n_moments: Number = 4
    h: Param = {'init':0.0, 'plim':(-0.5,0.5)}

    # Shape of the Voigt profile
    shape: Param = {'init':0.0, 'plim':(0.0,1.0)}

    profile: Literal[*(consts.LINE_PROFILES)] = 'gaussian'
    # profile: Literal[[prof.lower() for prof in line_profiles.keys()]] = 'gaussian'


class NarrowLine(BaseLine):
    type: Literal['narrow'] = 'narrow'


class BroadLine(BaseLine):
    type: Literal['broad'] = 'broad'
    disp: NonNegativeParam = {'init':500.0, 'plim':(300.0,3000.0)}
    voff: Param = {'init':0.0, 'plim':(-1000.0,1000.0), 'prior':{'type':'gaussian'}}


class AbsorpLine(BaseLine):
    type: Literal['absorp'] = 'absorp'


class CombinedLine(SpectralLine):
    type: Literal['combined'] = 'combined'
    children: list[Union['CombinedLine', 'NarrowLine', 'BroadLine', 'AbsorpLine']]


SpecLine = Annotated[NarrowLine | BroadLine | AbsorpLine | CombinedLine, Field()]


# TODO: add descriptions
class TestOptions(CustomBaseModel):
    mode: Literal['line', 'config'] = 'line'
    test_sets: list[list[dict] | dict] = Field(default_factory=list) # TODO: list[list[SpectralLine]]
    metrics: dict[str,Number] = Field(default_factory=dict)
    conv_mode: Literal['all', 'any'] = 'all'
    auto_stop: bool = False
    force_best: bool = True
    continue_fit: bool = True
    plot_tests: bool = True



class BadassConfig(CustomBaseModel):
    io: IOOptions = Field(default=IOOptions(), alias=AliasChoices('io','io_options'))
    fit: FitOptions = Field(default=FitOptions(), alias=AliasChoices('fit','fit_options'))
    mcmc: MCMCOptions = Field(default=MCMCOptions(), alias=AliasChoices('mcmc','mcmc_options'))
    comp: CompOptions = Field(default=CompOptions(), alias=AliasChoices('comp','comp_options'))

    power: PowerOptions = Field(default=PowerOptions(), alias=AliasChoices('power','power_options'))
    poly: PolyOptions = Field(default=PolyOptions(), alias=AliasChoices('poly','poly_options'))
    losvd: LOSVDOptions = Field(default=LOSVDOptions(), alias=AliasChoices('losvd','losvd_options'))
    host: HostOptions = Field(default=HostOptions(), alias=AliasChoices('host','host_options'))
    optfeii: VC04_Options | K10_Options = Field(default=VC04_Options(), discriminator='template', alias=AliasChoices('optfeii','opt_feii_options'))

    plot: PlotOptions = Field(default=PlotOptions(), alias=AliasChoices('plot', 'plot_options'))
    out: OutputOptions = Field(default=OutputOptions(), alias=AliasChoices('out', 'output_options'))

    narrow: NarrowLine = Field(default=NarrowLine(), alias=AliasChoices('narrow','narrow_options'))
    broad: BroadLine = Field(default=BroadLine(), alias=AliasChoices('broad','broad_options'))
    absorp: AbsorpLine = Field(default=AbsorpLine(), alias=AliasChoices('absorp','absorp_options'))

    test: TestOptions = Field(default=TestOptions(), alias=AliasChoices('test', 'test_options'))

    user_lines: list[SpecLine] = Field(default_factory=list) # TODO: | SpectralLine
    user_constraints: list[list[str | Number]] = Field(default_factory=list)
    user_mask: list[list[NonNegativeNum]] = Field(default_factory=list)

    _line_adapter = TypeAdapter(SpecLine)


    def add_line(self, line):
        self.user_lines.append(self._line_adapter.validate_python(line))


    def extend_lines(self, lines):
        self.user_lines.extend([self._line_adapter.validate_python(line) for line in lines])


    @classmethod
    def from_dict(cls, input_dict):
        print(input_dict)
        return cls(**input_dict)


    @classmethod
    def from_file(cls, _filepath):
        filepath = pathlib.Path(_filepath)
        if not filepath.exists():
            raise Exception('Unable to find options file: %s' % str(filepath))

        ext = filepath.suffix[1:]
        parse_func_name = 'parse_%s' % ext
        if not hasattr(cls, parse_func_name):
            raise Exception('Unsupported option file type: %s' % ext)

        return getattr(cls, parse_func_name)(filepath)


    # Custom file type parsers
    # Note: each parser should parse options to a dict and use
    #   BadassOptions.from_dict to initialize, allowing for
    #   option normalization and validation

    @classmethod
    def parse_json(cls, filepath):
        return cls.from_dict(json.load(filepath.open()))

    @classmethod
    def parse_py(cls, filepath):
        spec = importlib.util.spec_from_file_location('optmod', filepath)
        optmod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(optmod)
        return cls.from_dict({k:getattr(optmod, k) for k in dir(optmod) if not k[:2] == '__'})

    @classmethod
    def get_config(cls, options_data):
        if isinstance(options_data, list):
            return [cls.get_config(o) for o in options_data]

        if isinstance(options_data, dict):
            return cls.from_dict(options_data)

        if isinstance(options_data, pathlib.Path) or isinstance(options_data, str):
            return cls.from_file(options_data)

        return []

    @classmethod
    def get_config_from_args(cls, args):
        # function to handle the traditional way to call run_BADASS
        config_data = args.get('options', args.get('options_file', None))
        return cls.get_config(config_data)


def generate_opt_doc(name, opt, no_write=False):
    print('Generating docs for %s'%name)
    doc_str = ''

    # TODO: how to handle two templates?
    if name == 'optfeii':
        return


    doc_dir = pathlib.Path(__file__).parent.resolve().parent.parent.parent.joinpath('docs', 'documentation', 'usage')
    if opt.is_component:
        doc_dir = doc_dir.joinpath('components')

    if not opt.description is None:
        doc_str += opt.description.replace('\t','').replace('\n','').replace('  ','')
        doc_str += '\n'

    for field_name, field in opt.model_fields.items():

        req_str = ''
        extra = getattr(field, 'json_schema_extra', None)
        if (not extra is None) and (extra.get('required', False)):
            req_str = ' (Required)'

        def get_type(arg):
            if hasattr(arg, '__origin__'):
                return arg.__origin__.__name__
            if hasattr(arg, '__name__'):
                return arg.__name__
            return str(arg)


        if (isinstance(field.annotation, types.UnionType)) or (field.annotation.__name__ == 'Literal') or (field.annotation.__name__ == 'Union'):
            field_type = ' | '.join([get_type(arg) for arg in field.annotation.__args__])
        else:
            field_type = get_type(field.annotation)

        doc_str += '## `%s`%s\n'%(field_name, req_str)
        doc_str += '*Type:* `%s`<br/>\n'%field_type
        doc_str += '*Default:* `%s`<br/>\n'%str(field.default)

        if not field.description is None:
            doc_str += '*Description:* %s\n'%field.description

        if not field.examples is None:
            doc_str += '*Examples:* '
            doc_str += ', '.join(['`%s`'%str(ex) for ex in field.examples])
            doc_str += '\n'

        doc_str += '\n'

    print(doc_str)

    if no_write:
        return doc_str

    outname = name
    if not opt.alt_doc_name is None:
        outname = opt.alt_doc_name
    with open(doc_dir.joinpath('%s.md'%outname), 'w') as f:
        f.write(doc_str)


def generate_docs():
    # TODO: documentation for these
    ignore = ['user_lines', 'combined_lines', 'user_constraints', 'user_mask']

    for field_name, field in BadassConfig.model_fields.items():
        if field_name in ignore:
            continue
        generate_opt_doc(field_name, field.annotation)


if __name__ == '__main__':
    generate_docs()

