## `infmt` (Required)
*Type:* `sdss | muse | nirspec | miri | kcwi`<br/>
*Default:* `None`<br/>
*Description:* The format of the input file. Currently supported options: `['sdss', 'muse', 'nirspec', 'miri']`

## `output_dir`
*Type:* `Path`<br/>
*Default:* `None`<br/>
*Description:* The output directory of the BADASS results, logs, plots, etc.

## `overwrite`
*Type:* `bool`<br/>
*Default:* `False`<br/>
*Description:* If `True`, overwrite the `output_dir` if it already exists.

## `multiprocess`
*Type:* `bool`<br/>
*Default:* `False`<br/>
*Description:* For runs of multiple spectra or IFU cubes, run in multiprocess mode

## `log_level`
*Type:* `str`<br/>
*Default:* `info`<br/>
*Description:* The output log level. Options: `['debug', 'info', 'warning', 'error', 'critical']`

## `filter`
*Type:* `str`<br/>
*Default:* `None`<br/>
*Description:* The filter of the provided NIRSpec data cube.

## `disperser`
*Type:* `str`<br/>
*Default:* `None`<br/>
*Description:* The disperser of the provided NIRSpec data cube.

## `dust_cache`
*Type:* `Path`<br/>
*Default:* `None`<br/>
*Description:* Directory path to cache of Irsa dust extinction data.

