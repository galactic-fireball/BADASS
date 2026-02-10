The host model is used as a simplified placeholder in the event that the stellar continuum isn't of any interest. These are single-stellar population templates from the EMILES library, and do not have a low enough resolution for reliable stellar LOSVD fitting.
## `age`
*Type:* `list`<br/>
*Default:* `[0.1, 1.0, 10.0]`<br/>

## `amp`
*Type:* `dict | str | int | float`<br/>
*Default:* `{'init': '0.5*median_flux', 'plim': (0.0, 'max_flux')}`<br/>

## `vel`
*Type:* `dict | str | int | float`<br/>
*Default:* `{'init': 0.0, 'plim': (-500.0, 500.0)}`<br/>

## `disp`
*Type:* `dict | str | int | float`<br/>
*Default:* `{'init': 100.0, 'plim': (0.001, 500.0)}`<br/>

