## Under Construction

## NIRSpec

```python
options_file = OPTIONS_DIR.joinpath('nirspec_single.py')
test_file = EX_SPEC_DIR.joinpath('JWST_NIRSpec', 'NGC4051_nirspec_290_s3d.fits')
badass.run_BADASS(test_file, options_file=options_file)
```

where in `options_file` we have:

```python
io_options = {
	'infmt': 'nirspec',
    'output_dir': 'nirspec_out',
	'filter': '290',
	'disperser': 'h',
	'redshift': 0.002336,
	'spaxel': (25,25),
}

# Other fit options...
```
