import astropy.constants as const
import astropy.units as u
import pathlib
import prodict

DEFAULT_OUTDIR = 'MCMC_output'

# TODO: assign actual unit values via astropy?
c = const.c.to(u.km/u.s).value # speed of light (km/s)
k = const.k_B.to(u.m**2 * u.kg / u.s**2 / u.K).value # Boltzmann constant (m2 kg s-2 K-1)
h = const.h.to(u.m**2 * u.kg / u.s).value # Planck's constant (m2 kg / s)

GALACTIC_EBV = 0.04

# TODO: should be a fit_option?
MIN_FIT_REGION = 25 # (Angstroms), the minimum fitting region size

FIT_STATS = ['ML','OLS','RCHI2','RMSE']
LINE_PROFILES = ['gaussian','lorentzian','voigt','gauss-hermite','laplace','uniform']

LINE_TYPE_TO_NAME = {
    'na': 'narrow',
    'br': 'broad',
    'abs': 'absorp',
}

SUPPORTED_INSTRUMENTS = ['sdss', 'muse', 'nirspec', 'miri', 'kcwi', 'dja']

# For direct fitting of the stellar kinematics (stellar LOSVD), one can
# specify a stellar template library (Indo-US, Vazdekis 2010, or eMILES).
# One can also hold velocity or dispersion constant to avoid template
# convolution during the fitting process.
# Limits of the stellar template wavelength range
# The stellar templates packaged with BADASS are from the Indo-US Coude Feed Stellar Template Library
# with the below wavelength ranges.
LOSVD_LIBRARIES = prodict.Prodict.from_dict({
    'IndoUS' : {
        'fwhm_temp': 1.35,
        'min_losvd': 3460,
        'max_losvd': 9464,
    },
    'Vazdekis2010' : {
        'fwhm_temp': 2.51,
        'min_losvd': 3540.5,
        'max_losvd': 7409.6,
    },
    'eMILES' : {
        'fwhm_temp': 2.51,
        'min_losvd': 1680.2,
        'max_losvd': 49999.4,
    },
})


BADASS_DATA_DIR = pathlib.Path(__file__).parent.parent.joinpath('badass_data')
