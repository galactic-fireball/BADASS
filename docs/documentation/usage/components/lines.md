## Creating a line list

Strictly speaking, the only information BADASS needs is the center of a spectral feature. That said, it is recommended that you also provide a name for your own sanity.

Lines will be defined as "narrow" unless otherwise specified.

Many common AGN optical lines are already available via:

```python
import badass.components.spectral_lines.line_lists.common_lines as ba_lines
user_lines = [ba_lines.H_BETA,]
```

These lines are just python dictionaries that can be manipulated as needed.

To create your own lines:

```python
PA_ALPHA_LAM = (1.8751*u.um).to_value(u.AA) # wavelengths specified in angstroms

# single component narrow line:
PA_ALPHA = {'name': 'PA_ALPHA', 'center': PA_ALPHA_LAM,}
user_lines = [PA_ALPHA,]


# combined line with narrow and broad components:

# Note the children line components don't need to specify a center,
#they will inherit the one from the combined parent line
PA_ALPHA_NA = {'name': 'PA_ALPHA_NA',}

PA_ALPHA_BR = {
	'name': 'PA_ALPHA_BR',
	'type': 'broad',
	'voff': {'init': -100, 'plim': (-1000.0,0.0)},
	'voff_adjust': False,
}

PA_ALPHA = {
	'name': 'PA_ALPHA',
	'center': PA_ALPHA_LAM,
	'type': 'combined',
	'children':[PA_ALPHA_NA,PA_ALPHA_BR],
}

# we only need to specify the parent combined line here,
# BADASS will handle the children
user_lines = [PA_ALPHA,]

# Notice here that we updated the broad component's voff to make
# sure that it is negative (blueshifted).
# We also needed the 'voff_adjust': False so that BADASS doesn't
#readjust our 'init' value we set

# We can then set constraints to ensure that the broad component
# is indeed broader than the narrow:
user_constraints = [('PA_ALPHA_BR_DISP', 'PA_ALPHA_NA_DISP'),]
# each user constraint is specified with (a, b) where a >= b


```


#### Narrow Line Default Parameters (Gaussian)

| Parameter | Init                  | Limits           |
| --------- | --------------------- | ---------------- |
| amp       | `'0.001*median_flux'` | `(0.0,1.0)`      |
| disp      | `50.0`                | `(0.001,500.0)`  |
| voff      | `0.0`                 | `(-500.0,500.0)` |


#### Broad Line Default Parameters (Gaussian)

| Parameter | Init                  | Limits             |
| --------- | --------------------- | ------------------ |
| amp       | `'0.001*median_flux'` | `(0.0,1.0)`        |
| disp      | `500.0`               | `(300.0,3000.0)`   |
| voff      | `0.0`                 | `(-1000.0,1000.0)` |

