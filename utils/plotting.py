import astropy.constants as const
from astropy.stats import mad_std
import copy
import corner
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numexpr as ne
import numpy as np

import utils.utils as ba_utils

def calc_new_center(center, voff):
        return (voff*center)/const.c.to('km/s').value + center


def create_input_plot(ctx):
    fig = plt.figure(figsize=(14,8))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    fontsize = 16

    ### Un-normalized spectrum

    ax1.step(ctx.fit_wave, ctx.fit_spec*ctx.target.fit_norm, label='Object Fit Region', linewidth=0.5, color='xkcd:bright aqua')
    ax1.step(ctx.fit_wave, ctx.fit_noise*ctx.target.fit_norm, label=r'$1\sigma$ Uncertainty', linewidth=0.5, color='xkcd:bright orange')
    ax1.axhline(0.0, color='white', linewidth=0.5, linestyle='--')

    # TODO: change to masked_pixels
    if (hasattr(ctx.target, 'ibad')) and (len(ctx.target.ibad) > 0):
        for m in ibad:
            ax1.axvspan(ctx.fit_wave[m], ctx.fit_wave[m], alpha=0.25, color='xkcd:lime green')
        ax1.axvspan(0, 0, alpha=0.25, color='xkcd:lime green', label='bad pixels')

    ax1.set_title(r'Input Spectrum', fontsize=fontsize)
    ax1.set_xlabel(r'$\lambda_{\rm{rest}}$ ($\mathrm{\AA}$)', fontsize=fontsize)
    ax1.set_ylabel(r'$f_\lambda$ ($10^{%d}$ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$)' % (np.log10(ctx.options.fit_options.flux_norm)), fontsize=fontsize)
    ax1.set_xlim(np.min(ctx.fit_wave), np.max(ctx.fit_wave))
    ax1.legend(loc='best')

    ### Normalized spectrum

    ax2.step(ctx.fit_wave, ctx.fit_spec, label='Object Fit Region', linewidth=0.5, color='xkcd:bright aqua')
    ax2.step(ctx.fit_wave, ctx.fit_noise, label=r'$1\sigma$ Uncertainty', linewidth=0.5, color='xkcd:bright orange')
    ax2.axhline(0.0, color='white', linewidth=0.5, linestyle='--')

    # TODO: change to masked_pixels
    if (hasattr(ctx.target, 'ibad')) and (len(ctx.target.ibad) > 0):
        for m in ibad:
            ax1.axvspan(ctx.fit_wave[m], ctx.fit_wave[m], alpha=0.25, color='xkcd:lime green')
        ax1.axvspan(0, 0, alpha=0.25, color='xkcd:lime green', label='bad pixels')
    
    ax2.set_title(r'Fitted Spectrum', fontsize=fontsize)
    ax2.set_xlabel(r'$\lambda_{\rm{rest}}$ ($\mathrm{\AA}$)', fontsize=fontsize)
    ax2.set_ylabel(r'$\textrm{Normalized Flux}$', fontsize=fontsize)
    ax2.set_xlim(np.min(ctx.fit_wave),np.max(ctx.fit_wave))

    plt.tight_layout()
    plt.savefig(ctx.target.outdir.joinpath('input_spectrum.pdf'))
    plt.close(fig)


def create_test_plot(target, fit_results, label_A, label_B, test_title=None):

    test_A_fit = fit_results[label_A]
    test_A_comps = {key:val[0] for key,val in test_A_fit['mccomps'].items()}
    test_A_wave = test_A_comps['WAVE']
    test_B_fit = fit_results[label_B]
    test_B_comps = {key:val[0] for key,val in test_B_fit['mccomps'].items()}
    test_B_wave = test_B_comps['WAVE']

    fig = plt.figure(figsize=(14,11))
    gs = gridspec.GridSpec(9,1)
    test_A_axes = (fig.add_subplot(gs[0:3,0]), fig.add_subplot(gs[3:4,0]))
    test_B_axes = (fig.add_subplot(gs[5:8,0]), fig.add_subplot(gs[8:9,0]))
    gs.update(wspace=0.0, hspace=0.0)

    linewidth_default = 0.5
    linestyle_default = '-'

    order = len([p for p in test_B_fit['mcpars'] if p.startswith('APOLY_')]) - 1
    apoly_label = '%d%s-order Add Poly' % (order,'tsnrhtdd'[(order//10%10!=1)*(order%10<4)*order%10::4])
    order = len([p for p in test_B_fit['mcpars'] if p.startswith('MPOLY_')]) - 1
    mpoly_label = '%d%s-order Mult Poly' % (order,'tsnrhtdd'[(order//10%10!=1)*(order%10<4)*order%10::4])

    # Common values between tests
    # (label, key, color, linewidth, linestyle)
    plot_vals = [
        ('Data', 'DATA', 'white', linewidth_default, linestyle_default),
        ('Host/Stellar', 'HOST_GALAXY', 'xkcd:bright green', linewidth_default, linestyle_default),
        ('AGN Cont', 'POWER', 'xkcd:red', linewidth_default, '--'),
        (apoly_label, 'APOLY', 'xkcd:bright purple', linewidth_default, linestyle_default),
        (mpoly_label, 'MPOLY', 'xkcd:lavender', linewidth_default, linestyle_default),
        ('Narrow FeII', 'NA_OPT_FEII_TEMPLATE', 'xkcd:yellow', linewidth_default, linestyle_default),
        ('Broad FeII', 'BR_OPT_FEII_TEMPLATE', 'xkcd:orange', linewidth_default, linestyle_default),
        ('F-transition FeII', 'F_OPT_FEII_TEMPLATE', 'xkcd:yellow', linewidth_default, linestyle_default),
        ('S-transition FeII', 'F_OPT_FEII_TEMPLATE', 'xkcd:mustard', linewidth_default, linestyle_default),
        ('G-transition FeII', 'F_OPT_FEII_TEMPLATE', 'xkcd:orange', linewidth_default, linestyle_default),
        ('Z-transition FeII', 'F_OPT_FEII_TEMPLATE', 'xkcd:rust', linewidth_default, linestyle_default),
        ('UV Iron', 'UV_IRON_TEMPLATE', 'xkcd:bright purple', linewidth_default, linestyle_default),
        ('Balmer Continuum', 'BALMER_CONT', 'xkcd:bright green', linewidth_default, '--'),
        ('Model', 'MODEL', 'xkcd:bright red', 1.0, linestyle_default), # make last so it is on top of others
    ]

    for label, key, color, linewidth, linestyle in plot_vals:
        if (key not in test_A_comps) or (key not in test_B_comps):
            continue
        test_A_axes[0].plot(test_A_wave, test_A_comps[key], color=color, linewidth=linewidth, linestyle=linestyle, label=label)
        test_B_axes[0].plot(test_B_wave, test_B_comps[key], color=color, linewidth=linewidth, linestyle=linestyle, label=label)

    # {line_type: (label, color)}
    line_vals = {
        'na': ('Narrow/Core Comp', 'xkcd:cerulean'),
        'br': ('Broad Comp', 'xkcd:bright teal'),
        'abs': ('Absorption Comp', 'xkcd:pastel red'),
        'user': ('Other', 'xkcd:electric lime'),
    }

    # TODO: reduce dup code
    for line_name, line_dict in test_A_fit['line_list'].items():
        label, color = line_vals[line_dict['line_type']]
        test_A_axes[0].plot(test_A_wave, test_A_comps[line_name], color=color, linewidth=0.5, linestyle='-', label=label)

    for line_name, line_dict in test_B_fit['line_list'].items():
        label, color = line_vals[line_dict['line_type']]
        test_B_axes[0].plot(test_B_wave, test_B_comps[line_name], color=color, linewidth=0.5, linestyle='-', label=label)

    for comp_dict, ax in [(test_A_comps,test_A_axes),(test_B_comps,test_B_axes)]:
        ax[0].set_xticklabels([])
        ax[0].set_xlim(np.min(comp_dict['WAVE'])-10, np.max(comp_dict['WAVE'])+10)
        ax[0].set_ylabel('Normalized Flux',fontsize=10)

        sigma_resid = np.nanstd(comp_dict['DATA']-comp_dict['MODEL'])
        sigma_noise = np.nanmedian(comp_dict['NOISE'])
        ax[1].plot(comp_dict['WAVE'], comp_dict['NOISE']*3.0, linewidth=0.5, color='xkcd:bright orange', label=r'$\sigma_{\mathrm{noise}}=%0.4f$' % sigma_noise)
        ax[1].plot(comp_dict['WAVE'], comp_dict['RESID']*3.0, linewidth=0.5, color='white', label=r'$\sigma_{\mathrm{resid}}=%0.4f$' % sigma_resid)
        ax[1].axhline(0.0, linewidth=1.0, color='white', linestyle='--')

        ax_low = np.min([ax[0].get_ylim()[0], ax[1].get_ylim()[0]])
        ax_upp = np.max([ax[0].get_ylim()[1], ax[1].get_ylim()[1]])
        if np.isfinite(sigma_resid): ax_upp += 3.0 * sigma_resid

        minimum = np.nanmin([np.nanmin(vals) for vals in comp_dict.values()])
        if (not np.isfinite(minimum)) or (np.isnan(minimum)): minimum = 0.0

        ax[0].set_ylim(np.nanmin([0.0, minimum]), ax_upp)
        ax[0].set_xlim(np.min(comp_dict['WAVE']), np.max(comp_dict['WAVE']))
        ax[1].set_ylim(ax_low, ax_upp)
        ax[1].set_xlim(np.min(comp_dict['WAVE']), np.max(comp_dict['WAVE']))

        ax[1].set_yticklabels(np.round(np.array(ax[1].get_yticks()/3.0)))
        ax[1].set_ylabel(r'$\Delta f_\lambda$', fontsize=12)
        ax[1].set_xlabel(r'Wavelength, $\lambda\;(\mathrm{\AA})$', fontsize=12)

        handles, labels = ax[0].get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax[0].legend(unique_labels.values(), unique_labels.keys(), loc='upper right', fontsize=8)
        ax[1].legend(loc='upper right', fontsize=8)

    for test_label, comp_dict, ax in ((label_A, test_A_comps, test_A_axes), (label_B, test_B_comps, test_B_axes)):
        line_list = fit_results[test_label]['line_list']
        for line_name, line_dict in line_list.items():
            if 'label' not in line_dict:
                continue

            voff = fit_results[test_label]['mcpars'].get('%s_VOFF'%line_name, {}).get('med',np.nan)
            if voff == np.nan:
                continue

            xloc = calc_new_center(line_dict['center'], voff)
            idx = ba_utils.find_nearest(comp_dict['WAVE'], xloc)[1]
            yloc = np.max([comp_dict['DATA'][idx], comp_dict['MODEL'][idx]])*1.05

            ax[0].annotate(line_dict['label'], xy=(xloc,yloc), xycoords='data', xytext=(xloc,yloc), textcoords='data', horizontalalignment='center', verticalalignment='center', color='xkcd:white', fontsize=6)

    test_A_axes[0].set_title(r'$\textrm{TEST%s: %s}$'%(' '+test_title.replace('_', '\\_') if test_title else '', label_A), fontsize=16)
    test_B_axes[0].set_title(r'$\textrm{TEST%s: %s}$'%(' '+test_title.replace('_', '\\_') if test_title else '', label_B), fontsize=16)

    fig.tight_layout()
    plot_dir = target.outdir.joinpath('test_plots')
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_dir.joinpath('test%s_%s_vs_%s'%('_'+test_title if test_title else '', label_A, label_B)), bbox_inches='tight', dpi=300)
    plt.close()


def plot_ml_results(ctx):
    plot_best_model(ctx, 'max_likelihood_fit.pdf')
    if (not ctx.target.options.mcmc_options.mcmc_fit) and (ctx.target.options.plot_options.plot_HTML):
        plotly_best_fit(ctx)


def plot_best_model(ctx, plot_name):
    # TODO: need to copy? just let them be rescaled
    comp_dict = copy.deepcopy(ctx.comp_dict)
    for key in comp_dict:
        if key not in ['WAVE']:
            comp_dict[key] *= ctx.target.fit_norm

    fig = plt.figure(figsize=(14,6))
    gs = gridspec.GridSpec(4, 1)
    gs.update(wspace=0.0, hspace=0.0) # set the spacing between axes
    ax1 = plt.subplot(gs[0:3,0])
    ax2 = plt.subplot(gs[3,0])

    linewidth_default = 0.5
    linestyle_default = '-'

    ordinal = lambda n: '%d%s' % (n, 'tsnrhtdd'[(n//10%10!=1)*(n%10<4)*n%10::4])
    apoly_label = ordinal(len([p for p in ctx.fit_results.keys() if p.startswith('APOLY_')])-1)
    mpoly_label = ordinal(len([p for p in ctx.fit_results.keys() if p.startswith('MPOLY_')])-1)

    wave = comp_dict['WAVE']
    fit_mask = ctx.target.fit_mask

    # (label, key, color, linewidth, linestyle)
    plot_vals = [
       ('Data', 'DATA', 'white', linewidth_default, linestyle_default),
       ('Model', 'MODEL', 'xkcd:bright red', 1.0, linestyle_default),
       ('Host/Stellar', 'HOST_GALAXY', 'xkcd:bright green', linewidth_default, linestyle_default),
       ('AGN Cont.', 'POWER', 'xkcd:red', linewidth_default, '--'),
       ('%s-order Add. Poly.'%apoly_label, 'APOLY', 'xkcd:bright purple', linewidth_default, linestyle_default),
       ('%s-order Mult. Poly.'%mpoly_label, 'MPOLY', 'xkcd:lavender', linewidth_default, linestyle_default),
       ('Narrow FeII', 'NA_OPT_FEII_TEMPLATE', 'xkcd:yellow', linewidth_default, linestyle_default),
       ('Broad FeII', 'BR_OPT_FEII_TEMPLATE', 'xkcd:orange', linewidth_default, linestyle_default),
       ('F-transition FeII', 'F_OPT_FEII_TEMPLATE', 'xkcd:yellow', linewidth_default, linestyle_default),
       ('S-transition FeII', 'S_OPT_FEII_TEMPLATE', 'xkcd:mustard', linewidth_default, linestyle_default),
       ('G-transition FeII', 'G_OPT_FEII_TEMPLATE', 'xkcd:orange', linewidth_default, linestyle_default),
       ('Z-transition FeII', 'Z_OPT_FEII_TEMPLATE', 'xkcd:rust', linewidth_default, linestyle_default),
       ('UV Iron', 'UV_IRON_TEMPLATE', 'xkcd:bright purple', linewidth_default, linestyle_default),
       ('Balmer Continuum', 'BALMER_CONT', 'xkcd:bright green', linewidth_default, '--'),
    ]

    for label, key, color, linewidth, linestyle in plot_vals:
        if not key in comp_dict:
            continue
        ax1.plot(wave, comp_dict[key], color=color, linewidth=linewidth, linestyle=linestyle, label=label)

    # (label, color, linewidth, linestyle)
    line_params = {
        'na': ('Narrow/Core Comp.', 'xkcd:cerulean', linewidth_default, linestyle_default),
        'br': ('Broad Comp.', 'xkcd:bright teal', linewidth_default, linestyle_default),
        'out': ('Outflow Comp.', 'xkcd:pink', linewidth_default, linestyle_default),
        'abs': ('Absorption Comp.', 'xkcd:pastel red', linewidth_default, linestyle_default),
        'user': ('Other', 'xkcd:electric lime', linewidth_default, linestyle_default),
    }

    for line_name, line_dict in ctx.line_list.items():
        if not line_dict['line_type'] in line_params:
            continue
        if not line_name in comp_dict:
            continue # TODO: remove line from line_list
        label, color, linewidth, linestyle = line_params[line_dict['line_type']]
        ax1.plot(wave, comp_dict[line_name], color=color, linewidth=linewidth, linestyle=linestyle, label=label)


    ibad = [i for i in range(len(ctx.target.wave)) if i not in fit_mask]
    for m in ibad:
        ax1.axvspan(ctx.target.wave[m], ctx.target.wave[m], alpha=0.25, color='xkcd:lime green')
    ax1.axvspan(0, 0, alpha=0.25, color='xkcd:lime green', label='bad pixels')

    # Residuals
    sigma_resid = np.nanstd(comp_dict['DATA'][fit_mask]-comp_dict['MODEL'][fit_mask])
    sigma_noise = np.nanmedian(comp_dict['NOISE'][fit_mask])
    ax2.plot(ctx.target.wave, comp_dict['NOISE']*3.0, linewidth=0.5,color='xkcd:bright orange', label=r'$\sigma_{\mathrm{noise}}=%0.4f$' % sigma_noise)
    ax2.plot(ctx.target.wave, comp_dict['RESID']*3.0, linewidth=0.5,color='white', label=r'$\sigma_{\mathrm{resid}}=%0.4f$' % sigma_resid)
    ax1.axhline(0.0, linewidth=1.0, color='white', linestyle='--')
    ax2.axhline(0.0, linewidth=1.0, color='white', linestyle='--')

    # Axes limits
    ax_low = np.nanmin([ax1.get_ylim()[0], ax2.get_ylim()[0]])
    ax_upp = np.nanmax(comp_dict['DATA'][fit_mask])+(3.0 * np.nanmedian(comp_dict['NOISE'][fit_mask]))

    minimum = [np.nanmin(val[np.where(np.isfinite(val))[0]]) for comp, val in comp_dict.items() if val[np.isfinite(val)[0]].size > 0]
    minimum = np.nanmin(minimum) if len(minimum) > 0 else 0.0
    ax1.set_ylim(np.nanmin([0.0,minimum]), ax_upp)
    ax1.set_xlim(np.min(ctx.target.wave), np.max(ctx.target.wave))

    ax2.set_ylim(ax_low, ax_upp)
    ax2.set_xlim(np.min(ctx.target.wave), np.max(ctx.target.wave))

    # Axes labels
    ax1.set_xticklabels([])
    ax1.set_ylabel(r'$f_\lambda$ ($10^{-17}$ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$)', fontsize=10)
    ax2.set_yticklabels(np.round(np.array(ax2.get_yticks()/3.0)))
    ax2.set_ylabel(r'$\Delta f_\lambda$', fontsize=12)
    ax2.set_xlabel(r'Wavelength, $\lambda\;(\mathrm{\AA})$', fontsize=12)
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)

    # Gather up emission line center wavelengths and labels (if available, removing any duplicates)
    line_labels = []
    for line_name, line_dict in ctx.line_list.items():
        if (not 'label' in line_dict) or (line_dict['label'] in line_labels):
            continue
        label = line_dict['label']
        line_labels.append(label)

        # TODO: do this elsewhere for each line and store in results
        center = line_dict['center']
        if line_dict['voff'] == 'free':
            voff = ctx.cur_params[line_name+'_VOFF']
        else:
            voff = ne.evaluate(line_dict['voff'], local_dict=ctx.cur_params).item()
        xloc = calc_new_center(center, voff)
        offset_factor = 0.05
        wave_arg = ba_utils.find_nearest(ctx.target.wave, xloc)[1]
        yloc = np.max([comp_dict['DATA'][wave_arg], comp_dict['MODEL'][wave_arg]]) + (offset_factor*np.max(comp_dict['DATA']))
        ax1.annotate(label, xy=(xloc, yloc), xycoords='data', xytext=(xloc, yloc), textcoords='data',
                     horizontalalignment='center', verticalalignment='bottom', color='xkcd:white', fontsize=6)

    ax1.set_title(r'%s'%ctx.target.outdir.name.replace('_', '\\_'), fontsize=12)
    plt.savefig(ctx.target.outdir.joinpath(plot_name))
    plt.close()


def plotly_best_fit(ctx):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # TODO: need to copy? just let them be rescaled
    comp_dict = copy.deepcopy(ctx.comp_dict)
    for key in comp_dict:
        if key not in ['WAVE']:
            comp_dict[key] *= ctx.target.fit_norm
    wave = comp_dict['WAVE']

    fig = make_subplots(rows=2, cols=1, row_heights=(3,1))

    dash_default = 'solid'
    # (label, key, color, dash, legendrank)
    plot_vals = [
        ('Data', 'DATA', 'white', dash_default, 1),
        ('Model', 'MODEL', 'red', dash_default, 2),
        ('Noise', 'NOISE', '#FE00CE', dash_default, 3),
        ('Host Galaxy', 'HOST_GALAXY', 'lime', dash_default, 4),
        ('Power-law', 'POWER', 'red', 'dash', 5),
        ('Balmer cont.', 'BALMER_CONT', 'lime', 'dash', 6),
        ('UV Iron', 'UV_IRON_TEMPLATE', '#AB63FA', dash_default, 7),
        ('Narrow FeII', 'NA_OPT_FEII_TEMPLATE', 'rgb(255,255,51)', dash_default, 7),
        ('Broad FeII', 'BR_OPT_FEII_TEMPLATE', '#FF7F0E', dash_default, 8),
        ('F-transition FeII', 'F_OPT_FEII_TEMPLATE', 'rgb(255,255,51)', dash_default, 7),
        ('S-transition FeII', 'S_OPT_FEII_TEMPLATE', 'rgb(230,171,2)', dash_default, 8),
        ('G-transition FeII', 'G_OPT_FEII_TEMPLATE', '#FF7F0E', dash_default, 9),
        ('Z-transition FeII', 'Z_OPT_FEII_TEMPLATE', 'rgb(217,95,2)', dash_default, 10),
    ]

    for label, key, color, dash, legendrank in plot_vals:
        if not key in comp_dict:
            continue
        fig.add_trace(go.Scatter(x=wave, y=comp_dict[key], mode='lines', line=go.scatter.Line(color=color, width=1, dash=dash), name=label, legendrank=legendrank, showlegend=True), row=1, col=1)

    # (legendgroup, color, legendrank)
    line_params = {
        'na': ('narrow lines', '#00B5F7', 11),
        'br': ('broad lines', '#22FFA7', 13),
        'out': ('outflow lines', '#FC0080', 14),
        'abs': ('absorption lines', '#DA16FF', 15),
        'user': ('user lines', 'rgb(153,201,59)', 16),
    }

    for line_name, line_dict in ctx.line_list.items():
        if not line_dict['line_type'] in line_params:
            continue
        if not line_name in comp_dict:
            continue # TODO: remove line from line_list

        legendgroup, color, legendrank = line_params[line_dict['line_type']]
        fig.add_trace(go.Scatter(x=wave, y=comp_dict[line_name], mode='lines', line=go.scatter.Line(color=color, width=1), name=line_name, legendgroup=legendgroup, legendgrouptitle_text=legendgroup, legendrank=legendrank), row=1, col=1)

    fig.add_hline(y=0.0, line=dict(color='gray', width=2), row=1, col=1)  

    # Residuals
    fig.add_trace(go.Scatter(x=wave, y=comp_dict['RESID'], mode='lines', line=go.scatter.Line(color='white', width=1), name='Residuals', showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=wave, y=comp_dict['NOISE'], mode='lines', line=go.scatter.Line(color='#FE00CE', width=1), name='Noise', showlegend=False, legendrank=3,), row=2, col=1)

    fig.update_layout(
        autosize=False,
        width=1700,
        height=800,
        margin=dict(
            l=100,
            r=100,
            b=100,
            t=100,
            pad=1
        ),
        title= ctx.target.outdir.name,
        font_family='Times New Roman',
        font_size=16,
        font_color='white',
        legend_title_text='Components',
        legend_bgcolor='black',
        paper_bgcolor='black',
        plot_bgcolor='black',
    )

    fig.update_xaxes(title=r'$\Large\lambda_{\rm{rest}}\;\left[Å\right]$', linewidth=0.5, linecolor='gray', mirror=True, 
                     gridwidth=1, gridcolor='#222A2A', zerolinewidth=2, zerolinecolor='#222A2A', row=1, col=1)
    fig.update_xaxes(title=r'$\Large\lambda_{\rm{rest}}\;\left[Å\right]$', linewidth=0.5, linecolor='gray', mirror=True,
                     gridwidth=1, gridcolor='#222A2A', zerolinewidth=2, zerolinecolor='#222A2A', row=2, col=1)

    fig.update_yaxes(title=r'$\Large f_\lambda\;\left[\rm{erg}\;\rm{cm}^{-2}\;\rm{s}^{-1}\;Å^{-1}\right]$', linewidth=0.5, linecolor='gray',  mirror=True,
                     gridwidth=1, gridcolor='#222A2A', zerolinewidth=2, zerolinecolor='#222A2A', row=1, col=1)
    fig.update_yaxes(title=r'$\Large\Delta f_\lambda$', linewidth=0.5, linecolor='gray', mirror=True,
                     gridwidth=1, gridcolor='#222A2A', zerolinewidth=2, zerolinecolor='#222A2A', row=2, col=1)
        
    fig.update_xaxes(matches='x')
    fig.write_html(ctx.target.outdir.joinpath('%s_bestfit.html' % ctx.target.outdir.name), include_mathjax='cdn')


def posterior_plot(key, mcmc_results, chain, burn_in, outdir):
    # Plot posterior distributions and chains from MCMC.

    hist, bin_edges = np.histogram(mcmc_results['flat_chain'], bins='doane', density=False)
    # Generate pseudo-data on the ends of the histogram; this prevents the KDE from weird edge behavior
    n_pseudo = 3
    bin_width = bin_edges[1]-bin_edges[0]
    lower_pseudo_data = np.random.uniform(low=bin_edges[0]-bin_width*n_pseudo, high=bin_edges[0], size=hist[0]*n_pseudo)
    upper_pseudo_data = np.random.uniform(low=bin_edges[-1], high=bin_edges[-1]+bin_width*n_pseudo, size=hist[-1]*n_pseudo)
    h = ba_utils.kde_bandwidth(mcmc_results['flat_chain']) # Calculate bandwidth for KDE (Silverman method)

    # Create a subsampled grid for the KDE based on the subsampled data;
    # by default, we subsample by a factor of 10
    xs = np.linspace(np.min(mcmc_results['flat_chain']), np.max(mcmc_results['flat_chain']), 10*len(hist))
    kde = ba_utils.gauss_kde(xs, np.concatenate([mcmc_results['flat_chain'], lower_pseudo_data, upper_pseudo_data]), h)

    fig = plt.figure(figsize=(10,8))
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace=0.35, hspace=0.35)
    ax1 = plt.subplot(gs[0,0])
    ax2 = plt.subplot(gs[0,1])
    ax3 = plt.subplot(gs[1,0:2])

    # Plot 1: Histogram plots
    # 'Doane' binning produces the best results from tests
    n, bins, patches = ax1.hist(mcmc_results['flat_chain'], bins='doane', histtype='bar', density=True, facecolor='#4200a6', zorder=10)
    ax1.axvline(mcmc_results['best_fit'], linewidth=0.5, color='xkcd:bright aqua', zorder=20, label=r'$p(\theta|x)_{\rm{med}}$')
    ax1.axvline(mcmc_results['best_fit']-mcmc_results['ci_68_low'], linewidth=0.5, linestyle='--', color='xkcd:bright aqua', zorder=20, label=r'$\textrm{68\% conf.}$')
    ax1.axvline(mcmc_results['best_fit']+mcmc_results['ci_68_upp'], linewidth=0.5, linestyle='--', color='xkcd:bright aqua', zorder=20)
    ax1.axvline(mcmc_results['best_fit']-mcmc_results['ci_95_low'], linewidth=0.5, linestyle=':', color='xkcd:bright aqua', zorder=20, label=r'$\textrm{95\% conf.}$')
    ax1.axvline(mcmc_results['best_fit']+mcmc_results['ci_95_low'], linewidth=0.5, linestyle=':', color='xkcd:bright aqua', zorder=20)

    ax1.plot(xs, kde, linewidth=0.5, color='xkcd:bright pink', zorder=15, label='KDE')
    ax1.plot(xs, kde, linewidth=3.0, color='xkcd:bright pink', alpha=0.50, zorder=15)
    ax1.plot(xs, kde, linewidth=6.0, color='xkcd:bright pink', alpha=0.20, zorder=15)

    ax1.grid(visible=True, which='major', axis='both', alpha=0.15, color='xkcd:bright pink', linewidth=0.5, zorder=0)
    ax1.set_xlabel(r'%s' % key, fontsize=12)
    ax1.set_ylabel(r'$p$(%s)' % key, fontsize=12)
    ax1.legend(fontsize=6)
    
    # Plot 2: best fit values
    values_dict = {
        'best_fit': r'$p(\theta|x)_{\rm{med}}$',
        'ci_68_low': r'$\rm{CI\;68\%\;low}$', 'ci_68_upp': r'$\rm{CI\;68\%\;upp}$',
        'ci_95_low': r'$\rm{CI\;95\%\;low}$', 'ci_95_upp': r'$\rm{CI\;95\%\;upp}$',
        'mean': r'$\rm{Mean}$', 'std_dev': r'$\rm{Std.\;Dev.}$',
        'median': r'$\rm{Median}$', 'med_abs_dev': r'$\rm{Med. Abs. Dev.}$',
    }

    start, step = 1, 0.12
    vspace = np.linspace(start, 1-len(values_dict)*step, len(values_dict), endpoint=False)
    for i, (value, label) in enumerate(values_dict.items()):
        ax2.annotate('{0:>30}{1:<2}{2:<30.3f}'.format(label, r'$\qquad=\qquad$', mcmc_results[value]),
                    xy=(0.5, vspace[i]),  xycoords='axes fraction',
                    xytext=(0.95, vspace[i]), textcoords='axes fraction',
                    horizontalalignment='right', verticalalignment='top', fontsize=10)
    ax2.axis('off')

    # Plot 3: Chain plot
    nwalkers, niters = chain.shape
    for w in range(nwalkers):
        ax3.plot(range(niters), chain[w,:], color='white', linewidth=0.5, alpha=0.5, zorder=0)

    # Calculate median and median absolute deviation of walkers at each iteration; we have depreciated
    # the average and standard deviation because they do not behave well for outlier walkers, which
    # also don't agree with histograms
    c_med = np.nanmedian(chain, axis=0)
    c_madstd = mad_std(chain)

    ax3.plot(range(niters), c_med, color='xkcd:bright pink', linewidth=2.0, label='Median', zorder=10)
    ax3.fill_between(range(niters), c_med+c_madstd, c_med-c_madstd, color='#4200a6', alpha=0.5, linewidth=1.5, label='Median Absolute Dev.', zorder=5)
    ax3.axvline(burn_in, linestyle='--', linewidth=0.5, color='xkcd:bright aqua', label='burn-in = %d' % burn_in, zorder=20)
    ax3.grid(visible=True, which='major', axis='both', alpha=0.15, color='xkcd:bright pink', linewidth=0.5, zorder=0)
    ax3.set_xlim(0,niters)
    ax3.set_xlabel(r'$N_\mathrm{iter}$', fontsize=12)
    ax3.set_ylabel(r'%s' % key, fontsize=12)
    ax3.legend(loc='upper left')

    histo_dir = outdir.joinpath('histogram_plots')
    histo_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(histo_dir.joinpath('%s_MCMC.png' % (key)), bbox_inches='tight', dpi=300)
    plt.close()


def corner_plot(ctx):
    # create a corner plot of all or selected parameters

    flat_chains = ctx.mcmc_result_chains['flat_chains']
    plot_pars = [par for par in ctx.options.plot_options.corner_options.pars if par in flat_chains]
    if len(plot_pars) < 2: plot_pars = list(ctx.param_dict.keys()) # Default to free params

    if len(ctx.options.plot_options.corner_options.labels) == len(plot_pars):
        labels = ctx.options.plot_options.corner_options.labels
    else:
        labels = plot_pars

    flat_samples = np.vstack([flat_chains[k] for k in plot_pars]).T

    with plt.style.context('default'):
        fig = corner.corner(flat_samples, labels=labels)
        plt.savefig(ctx.target.outdir.joinpath('corner.pdf'))
        plt.close()

