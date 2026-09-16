import astropy.units as u

# TODO: line centers as any unit
PA_ALPHA_LAM = (1.8751*u.um).to(u.AA).value
NA_PA_ALPHA = {'name': 'NA_PA_ALPHA', 'center': PA_ALPHA_LAM,}
NA_PA_ALPHA_2 = {'name': 'NA_PA_ALPHA_2', 'center': PA_ALPHA_LAM,}
BR_PA_ALPHA = {'name': 'BR_PA_ALPHA', 'center': PA_ALPHA_LAM, 'type': 'broad',}
PA_ALPHA = {'name': 'PA_ALPHA', 'center': PA_ALPHA_LAM, 'type': 'combined', 'children': [NA_PA_ALPHA, NA_PA_ALPHA_2, BR_PA_ALPHA,],}
