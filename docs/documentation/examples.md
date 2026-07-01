## Under Construction

## Quick Start

```python
import badass

spec_file # a string path or a pathlib.Path
options # the path (string or pathlib) to a .py or .json file; or a python dict

badass.run_BADASS(spec_file, options=options)
```

## SDSS Simple Example

```python
def sdss_simple_example():

    cfg = {
        'io': {
            'infmt': 'sdss',
            'output_dir': 'sdss_example',
        },
        'fit': {
            'fit_reg': (4700, 5200),
        }
    }

    spec_file = EX_SPEC_DIR.joinpath('5-test', 'spec-0519-52283-0280.fits')
    badass.run_BADASS(spec_file, options=cfg)
```


## SDSS Example with Emission Lines

```python
def sdss_lines_example():

    from badass.components.spectral_lines.line_lists.common_lines import H_BETA, OIII_4960, OIII_5007

    cfg = {
        'io': {
            'infmt': 'sdss',
            'output_dir': 'sdss_example',
        },
        'fit': {
            'fit_reg': (4700, 5200),
        },
        'user_lines': [H_BETA, OIII_4960, OIII_5007],
    }

    spec_file = EX_SPEC_DIR.joinpath('5-test', 'spec-0519-52283-0280.fits')
    badass.run_BADASS(spec_file, options=cfg)

```


## NIRSpec Aperture Example

```python
def nirspec_aperture_example():
    cfg = {
        'io': {
            'infmt': 'nirspec',
            'output_dir': 'nirspec_example',
            'filter': '290',
            'disperser': 'h',
        },
        'fit': {
            'fit_reg': (36400,40000),
            'redshift': 0.002336,
            'fit_area': {
                'type':'aperture',
                'apertures': [{'shape':'circular', 'center':(30,27), 'radius':8},],
                'plot_input':True,
            },
        },
        'comp': {
            'fit_losvd': False,
            'fit_feii': False,
        },
    }

    spec_file = EX_SPEC_DIR.joinpath('JWST_NIRSpec', 'NGC4051_nirspec_290_s3d.fits')
    badass.run_BADASS(spec_file, options=cfg)

```

