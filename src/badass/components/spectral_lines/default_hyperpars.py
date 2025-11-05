type_default_hyperpars = {
	'NARROW': {
		'VOFF': {
			'INIT': 0.0,
			'PLIM': (-500,500),
		},
		'DISP': {
			'INIT': 50.0,
			'PLIM': (0.001,300.0),
		},
	},
	'BROAD': {
		'VOFF': {
			'INIT': 0.0,
			'PLIM': (-1000,1000),
		},
		'DISP': {
			'INIT': 500.0,
			'PLIM': (300.0,3000.0),
		},
	},
	'ABSORP': {
		'VOFF': {
			'INIT': 0.0,
			'PLIM': (-500,500),
		},
		'DISP': {
			'INIT': 50.0,
			'PLIM': (0.001,300.0),
		},
	},
	'COMMON': {
		'SHAPE': {
			'INIT': 0.0,
			'PLIM': (0.0,1.0),
		},
	},
}

type_default_hyperpars['COMMON'].update({'H%d'%h:{'INIT':0.0,'PLIM':(0.0,1.0),} for h in range(3,11)})


profile_default_hyperpars = {
	'LAPLACE': {
		'H3': {
			'INIT': 0.01,
			'PLIM': (-0.15,0.15),
		},
		'H4': {
			'INIT': 0.01,
			'PLIM': (0.0,0.2),
		},
	},
	'UNIFORM': {
		'H4': {
			'INIT': -0.01,
			'PLIM': (-0.3,-1e-4),
		},
	},
}
