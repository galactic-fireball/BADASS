import astropy.units as u
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import pathlib

from spark.io import load_spectral_product
from spark.plot import add_ax_labels

import badass


TESTING_DIR = pathlib.Path(__file__).resolve().parent
BADASS_DIR = TESTING_DIR.parent
EX_SPEC_DIR = BADASS_DIR.joinpath('examples', 'example_spectra')


x3C_305_S3D = EX_SPEC_DIR.joinpath('JWST_NIRSpec', '3C-305_nirspec_170_s3d.fits')
x3C_305_REDSHIFT = 0.041706


def data_cube_test():
    cube = load_spectral_product(x3C_305_S3D, 'NIRSPEC_IFU', z=x3C_305_REDSHIFT)
    med_cube = cube.get_median_map()
    center = cube.get_brightest_spaxel()
    print(center.coord)

    rad_as = 1.0 * u.arcsec
    rad_px = rad_as / cube.pix_scale
    print(rad_px)
    ap = cube.aperture('circular', center, radius=rad_px)

    fig, ax = plt.subplots()
    ax.imshow(med_cube.value, origin='lower', cmap='ocean', norm=LogNorm())
    ax.scatter(center.x, center.y, color='red', marker='x', s=50)
    ap.add_to_plot(ax, color='red')
    plt.show()

    fig, ax = plt.subplots()
    ap.add_spec_plot(ax, color='black')
    add_ax_labels(ax, 'um')
    plt.show()


def nirspec_demo():
    wave_pad = (0.2*u.um).to_value(u.AA)
    from badass.components.spectral_lines.line_lists.nir_hi import BR_PA_ALPHA, NA_PA_ALPHA, PA_ALPHA

    #### Line options:
    # lines = [PA_ALPHA,] # includes 2 narrow components and a broad component
    # lines = [NA_PA_ALPHA] # just the narrow component

    PA_ALPHA['children'] = [NA_PA_ALPHA, BR_PA_ALPHA]
    lines = [PA_ALPHA,] # includes 1 narrow and 1 broad component as specified in the previous line


    #### Fit Region options:

    fit_area_opts = {
        'type': 'aperture',
        'apertures': [
                # radius in pixels
                {'shape':'circular', 'center':(39,41), 'radius':10},
        ],
        'plot_input': False,
    }

    # fit_area_opts = {
    #     'type': 'spaxels',
    #     'spaxels': {'x':(37,40), 'y':(39,42),},
    #     'plot_input': False,
    # }

    fit_mask = [(1.95,1.975),]
    fit_mask = [((m[0]*u.um).to_value(u.AA), (m[1]*u.um).to_value(u.AA)) for m in fit_mask]

    ba_opts = {
        'io': {
            'infmt': 'nirspec',
            'output_dir': 'nirspec_demo',
            'filter': '170',
            'disperser': 'h',
            'plots': {'style': 'light',},
            # 'nprocesses': 2,
        },
        'fit': {
            # fit_reg in angstroms
            'fit_reg': (PA_ALPHA['center']-wave_pad, PA_ALPHA['center']+wave_pad),
            'redshift': x3C_305_REDSHIFT,
            'fit_area': fit_area_opts,
        },
        'comp': {
            'fit_host': False,
            'fit_power': True,
        },
        'mcmc': {
            'mcmc_fit': True,
            'max_iter': 300,
        },
        'user_lines': lines,
        'user_mask': fit_mask,
    }

    badass.run_BADASS(x3C_305_S3D, options_file=ba_opts)


def parameter_map_plot():
    from astropy.io import fits
    hdu = fits.open('nirspec_demo/3C-305_nirspec_170_s3d/parameter_maps.fits')
    print([t.name for t in hdu])
    # ['PRIMARY', 'POWER_AMP', 'POWER_SLOPE', 'NA_PA_ALPHA_AMP', 'NA_PA_ALPHA_VOFF', 'NA_PA_ALPHA_DISP', 'LOG_LIKE', 'R_SQUARED', 'RCHI2', 'NA_PA_ALPHA_FLUX', 'NA_PA_ALPHA_LUM', 'NA_PA_ALPHA_EW', 'NA_PA_ALPHA_FWHM', 'NA_PA_ALPHA_W80', 'NA_PA_ALPHA_NPIX', 'NA_PA_ALPHA_SNR']

    flux = 10**hdu['NA_PA_ALPHA_FLUX'].data
    fig, ax = plt.subplots()
    ax.imshow(flux, origin='lower')
    plt.show()


def result_test():
    from badass.runner.bootstrap import MLResult
    from badass.runner.mcmc import MCMCResult

    output_dir = TESTING_DIR.parent.joinpath('nirspec_demo', '3C-305_nirspec_170_s3d', 'ap_0')

    # res = MLResult.from_output(output_dir)
    res = MCMCResult.from_output(output_dir)

    res.dump()
    res.quick_view()


def main():
    # data_cube_test()
    # nirspec_demo()
    # parameter_map_plot()
    result_test()


if __name__ == '__main__':
    main()
