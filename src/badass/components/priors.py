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


def plot_priors():
    import matplotlib.pyplot as plt

    # a,b = 0.0001, 1.0
    a,b = 0.01, 1.25

    fig, ax = plt.subplots()
    x = np.linspace(stats.loguniform.ppf(0.01, a, b), stats.loguniform.ppf(0.99, a, b), 100)

    # ax.plot(x, stats.norm.pdf(x), lw=2, label='norm pdf')
    ax.plot(x, stats.norm.logpdf(x), lw=2, label='norm logpdf')

    # ax.plot(x, stats.loguniform.pdf(x, a, b), lw=2, label='loguniform pdf')
    ax.plot(x, stats.loguniform.logpdf(x, a, b), lw=2, label='loguniform logpdf')

    # ax.plot(x, stats.halfnorm.pdf(x), lw=2, label='halfnorm pdf')
    ax.plot(x, stats.halfnorm.logpdf(x), lw=2, label='halfnorm logpdf')

    ax.set_xlim(x[0],x[-1])
    ax.legend()

    plt.tight_layout()
    plt.show()


def main():
    plot_priors()


if __name__ == '__main__':
    main()
