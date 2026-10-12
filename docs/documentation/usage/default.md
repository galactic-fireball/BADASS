```json
{
  "io": {
    "infmt": null,
    "output_dir": null,
    "product_name": null,
    "overwrite": false,
    "nprocesses": 1,
    "out_fmt": "fits",
    "verbose_out": false,
    "log_level": "info",
    "filter": null,
    "grating": null,
    "disperser": null,
    "dust_cache": null,
    "plots": {
      "style": "dark",
      "best_model": true
    },
    "outputs": {
      "print_results": true
    }
  },
  "fit": {
    "fit_reg": "auto",
    "redshift": 0.0,
    "log_rebin_oversample": 1,
    "skip_bootstrap": false,
    "n_basinhop": 25,
    "max_like_niter": 10,
    "fit_area": {
      "type": null,
      "spaxels": null,
      "bins": {
        "side_length": 0,
        "method": "sum",
        "x": [
          0,
          -1
        ],
        "y": [
          0,
          -1
        ]
      },
      "apertures": null,
      "plot_input": false
    },
    "reweighting": false,
    "fit_stat": "ML",
    "cosmology": {
      "H0": 70.0,
      "Om0": 0.3,
      "Ode0": 0.7
    },
    "test_models": false,
    "mask_emline": false,
    "mask_bad_pix": false,
    "mask_metal": false,
    "feature_edge_pad": 10
  },
  "mcmc": {
    "mcmc_fit": false,
    "nwalkers": 100,
    "save_chains": false,
    "auto_stop": true,
    "conv_type": "median",
    "min_samp": 1000,
    "ncor_times": 10,
    "autocorr_tol": 10,
    "write_iter": 100,
    "write_thresh": 100,
    "burn_in": 1500,
    "min_iter": 100,
    "max_iter": 2500
  },
  "comp": {
    "fit_feii": false,
    "fit_uv_iron": false,
    "fit_balmer": false,
    "fit_losvd": false,
    "fit_host": true,
    "fit_power": false,
    "fit_poly": false,
    "fit_narrow": true,
    "fit_broad": true,
    "fit_absorp": false,
    "tie_line_disp": false,
    "tie_line_voff": false
  },
  "power": {
    "type": "simple",
    "amp": {
      "init": "0.5*median_flux",
      "plim": [
        0,
        "max_flux"
      ]
    },
    "slope": {
      "init": -1.0,
      "plim": [
        -6.0,
        6.0
      ]
    },
    "break_": {
      "init": "max_wave - 0.5*(max_wave-min_wave)",
      "plim": [
        "min_wave",
        "max_wave"
      ]
    },
    "slope_1": {
      "init": -1.0,
      "plim": [
        -6.0,
        6.0
      ]
    },
    "slope_2": {
      "init": -1.0,
      "plim": [
        -6.0,
        6.0
      ]
    },
    "curvature": {
      "init": 0.1,
      "plim": [
        0.01,
        1.0
      ]
    }
  },
  "poly": {
    "apoly_order": 0,
    "apoly_coeff": {
      "init": 0.0,
      "plim": [
        -100.0,
        100.0
      ]
    },
    "mpoly_order": 0,
    "mpoly_coeff": {
      "init": 0.0,
      "plim": [
        -100.0,
        100.0
      ]
    }
  },
  "losvd": {
    "library": "IndoUS",
    "vel": {
      "init": 100.0,
      "plim": [
        -500.0,
        500.0
      ]
    },
    "disp": {
      "init": 150.0,
      "plim": [
        0.001,
        500.0
      ]
    }
  },
  "host": {
    "age": [
      0.1,
      1.0,
      10.0
    ],
    "amp": {
      "init": "0.5*median_flux",
      "plim": [
        0.0,
        "max_flux"
      ]
    },
    "vel": {
      "init": 0.0,
      "plim": [
        -500.0,
        500.0
      ]
    },
    "disp": {
      "init": 100.0,
      "plim": [
        0.001,
        500.0
      ]
    }
  },
  "optfeii": {
    "template": "VC04",
    "na_amp": {
      "init": "0.1*median_flux",
      "plim": [
        0,
        "max_flux"
      ]
    },
    "br_amp": {
      "init": "0.1*median_flux",
      "plim": [
        0,
        "max_flux"
      ]
    },
    "na_disp": {
      "init": 10.0,
      "plim": [
        0.1,
        250.0
      ]
    },
    "br_disp": {
      "init": 500.0,
      "plim": [
        100,
        5000.0
      ]
    },
    "na_voff": {
      "init": 0.0,
      "plim": [
        -1000.0,
        1000.0
      ]
    },
    "br_voff": {
      "init": 0.0,
      "plim": [
        -2000.0,
        2000.0
      ]
    }
  },
  "narrow": {
    "name": "",
    "type": "narrow",
    "center": null,
    "amp": {
      "init": "0.001*median_flux",
      "plim": [
        0.0,
        1.0
      ]
    },
    "amp_adjust": true,
    "disp": {
      "init": 50.0,
      "plim": [
        0.001,
        500.0
      ]
    },
    "voff": {
      "init": 0.0,
      "plim": [
        -500.0,
        500.0
      ],
      "prior": {
        "type": "gaussian"
      }
    },
    "voff_adjust": true,
    "n_moments": 4,
    "h": {
      "init": 0.0,
      "plim": [
        -0.5,
        0.5
      ]
    },
    "shape": {
      "init": 0.0,
      "plim": [
        0.0,
        1.0
      ]
    },
    "profile": "gaussian"
  },
  "broad": {
    "name": "",
    "type": "broad",
    "center": null,
    "amp": {
      "init": "0.001*median_flux",
      "plim": [
        0.0,
        1.0
      ]
    },
    "amp_adjust": true,
    "disp": {
      "init": 500.0,
      "plim": [
        300.0,
        3000.0
      ]
    },
    "voff": {
      "init": 0.0,
      "plim": [
        -1000.0,
        1000.0
      ],
      "prior": {
        "type": "gaussian"
      }
    },
    "voff_adjust": true,
    "n_moments": 4,
    "h": {
      "init": 0.0,
      "plim": [
        -0.5,
        0.5
      ]
    },
    "shape": {
      "init": 0.0,
      "plim": [
        0.0,
        1.0
      ]
    },
    "profile": "gaussian"
  },
  "absorp": {
    "name": "",
    "type": "absorp",
    "center": null,
    "amp": {
      "init": "0.001*median_flux",
      "plim": [
        0.0,
        1.0
      ]
    },
    "amp_adjust": true,
    "disp": {
      "init": 50.0,
      "plim": [
        0.001,
        500.0
      ]
    },
    "voff": {
      "init": 0.0,
      "plim": [
        -500.0,
        500.0
      ],
      "prior": {
        "type": "gaussian"
      }
    },
    "voff_adjust": true,
    "n_moments": 4,
    "h": {
      "init": 0.0,
      "plim": [
        -0.5,
        0.5
      ]
    },
    "shape": {
      "init": 0.0,
      "plim": [
        0.0,
        1.0
      ]
    },
    "profile": "gaussian"
  },
  "test": {
    "mode": "config",
    "test_sets": [],
    "metrics": {},
    "conv_mode": "all",
    "auto_stop": false,
    "force_best": true,
    "continue_fit": true,
    "plot_tests": true
  },
  "user_lines": [],
  "user_constraints": [],
  "user_mask": []
}
```