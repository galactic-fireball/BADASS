## `type`
*Type:* `simple | broken`<br/>
*Default:* `simple`<br/>
*Examples:* `'simple'`, `'broken'`

## `amp`
*Type:* `dict | str | int | float`<br/>
*Default:* `{'init': '0.5*median_flux', 'plim': (0, 'max_flux')}`<br/>

## `slope`
*Type:* `dict | str | int | float`<br/>
*Default:* `{'init': -1.0, 'plim': (-6.0, 6.0)}`<br/>

## `break_`
*Type:* `dict | str | int | float`<br/>
*Default:* `{'init': 'max_wave - 0.5*(max_wave-min_wave)', 'plim': ('min_wave', 'max_wave')}`<br/>

## `slope_1`
*Type:* `dict | str | int | float`<br/>
*Default:* `{'init': -1.0, 'plim': (-6.0, 6.0)}`<br/>

## `slope_2`
*Type:* `dict | str | int | float`<br/>
*Default:* `{'init': -1.0, 'plim': (-6.0, 6.0)}`<br/>

## `curvature`
*Type:* `dict | str | int | float`<br/>
*Default:* `{'init': 0.1, 'plim': (0.01, 1.0)}`<br/>

