import astropy.constants as const
import copy
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numexpr as ne
import numpy as np

from utils.utils import find_nearest

def plot_ml_results(ctx):
    max_like_plot(ctx)
    if (not ctx.target.options.mcmc_options.mcmc_fit) and (ctx.target.options.plot_options.plot_HTML):
        plotly_best_fit(ctx)


def max_like_plot(ctx):
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

    ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])
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

    # Emission line annotations
    def calc_new_center(center, voff):
        return (voff*center)/const.c.to('km/s').value + center

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
        wave_arg = find_nearest(ctx.target.wave, xloc)[1]
        yloc = np.max([comp_dict['DATA'][wave_arg], comp_dict['MODEL'][wave_arg]]) + (offset_factor*np.max(comp_dict['DATA']))
        ax1.annotate(label, xy=(xloc, yloc), xycoords='data', xytext=(xloc, yloc), textcoords='data',
                     horizontalalignment='center', verticalalignment='bottom', color='xkcd:white', fontsize=6)

    ax1.set_title(r'%s'%ctx.target.outdir.name.replace('_', '\\_'), fontsize=12)
    plt.savefig(ctx.target.outdir.joinpath('max_likelihood_fit.pdf'))
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

