## `infmt` (Required)
*Type:* `sdss | muse | nirspec | dja | miri | kcwi`<br/>
*Default:* `None`<br/>
*Description:* The format of the input file. Currently supported options: `['sdss', 'muse', 'nirspec', 'dja', 'miri']`

## `output_dir`
*Type:* `Path`<br/>
*Default:* `None`<br/>
*Description:* The output directory of the BADASS results, logs, plots, etc.

## `overwrite`
*Type:* `bool`<br/>
*Default:* `False`<br/>
*Description:* If `True`, overwrite the `output_dir` if it already exists.

## `nprocesses`
*Type:* `int`<br/>
*Default:* `1`<br/>
*Description:* The number of fitting processes to run at once. Provide > 1 for multiprocessing.

## `log_level`
*Type:* `str`<br/>
*Default:* `info`<br/>
*Description:* The output log level. Options: `['debug', 'info', 'warning', 'error', 'critical']`

## `filter`
*Type:* `str`<br/>
*Default:* `None`<br/>
*Description:* The filter of the provided NIRSpec data cube. This will be otherwise auto-found from the fits header.

## `disperser`
*Type:* `str`<br/>
*Default:* `None`<br/>
*Description:* The disperser of the provided NIRSpec data cube. This will be otherwise auto-found from the fits header.

## `dust_cache`
*Type:* `Path`<br/>
*Default:* `None`<br/>
*Description:* Directory path to cache of Irsa dust extinction data.

