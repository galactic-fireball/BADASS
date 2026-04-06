import numpy as np
from scipy import stats

def lnprior_gaussian(x, **kwargs):
    """
    Log-Gaussian prior based on user-input. If not specified, mu and sigma
    will be derived from the init and plim, with plim occurring at 5-sigma
    for the maximum plim from the mean.
    """
    sigma_level = 5
    loc = kwargs['prior'].get('loc', kwargs['init'])
    scale = kwargs['prior'].get('scale', np.max(np.abs(kwargs['plim']))/sigma_level)
    return stats.norm.logpdf(x, loc=loc, scale=scale)


def lnprior_halfnorm(x, **kwargs):
    """
    Half Log-Normal prior based on user-input. If not specified, mu and sigma
    will be derived from the init and plim, with plim occurring at 5-sigma
    for the maximum plim from the mean.
    """
    sigma_level = 5
    x = np.abs(x)
    loc = kwargs['prior'].get('loc', kwargs['plim'][0])
    scale = kwargs['prior'].get('scale', np.max(np.abs(kwargs['plim']))/sigma_level)
    return stats.halfnorm.logpdf(x, loc=loc, scale=scale)


def lnprior_jeffreys(x, **kwargs):
    """
    Log-Jeffreys prior based on user-input.  If not specified, mu and sigma
    will be derived from the init and plim, with plim occurring at 5-sigma
    for the maximum plim from the mean.
    """
    x = np.abs(x)
    if np.any(x) <= 0: x = 1.e-6
    scale = 1
    if 'loc' in kwargs['prior']:
        loc = np.abs(kwargs['prior']['loc'])
    else:
        loc = np.min(np.abs(kwargs['plim']))
    a, b = np.min(np.abs(kwargs['plim'])), np.max(np.abs(kwargs['plim']))
    if a <= 0: a = 1e-6
    return stats.loguniform.logpdf(x, a=a, b=b, loc=loc, scale=scale)


def lnprior_flat(x, **kwargs):
    if (x >= kwargs['plim'][0]) and (x <= kwargs['plim'][1]):
        return 1.0
    return -np.inf
