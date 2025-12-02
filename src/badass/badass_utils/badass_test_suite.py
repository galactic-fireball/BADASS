import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import chisquare, f, median_abs_deviation, norm

plt.style.use('dark_background')
plt.rcParams['text.usetex'] = True


def r_squared(data, model):
    # Simple calculation of R-squared statistic for a single fit

    # Calculate residual sum-of-squares (RSS)
    rss = np.nansum((data-model)**2)
    # Calculate total sum-of-squares (TSS)
    tss = np.nansum((data)**2)
    return 1-rss/tss


def r_chi_squared(data, model, noise, npar):
    # Simple calculation of reduced Chi-squared statistic for a single fit

    # Degrees of freedom (number of data minus free fitted parameters)
    nu = len(data)-npar
    rchi2 = np.nansum((data-model)**2/noise**2)/nu
    return rchi2


def root_mean_squared_error(data, model):
    # Simple calculation of root mean squared error (RMSE) statistic for a single fit

    # Normalize
    data_med = np.nanmedian(data)
    data = data / data_med
    model = model / data_med
    return np.sqrt(1.0/len(data) * np.nansum((data-model)**2))


def mean_abs_error(data, model):
    # Simple calculation of mean absolute error (MAE) statistic for a single fit

    # Normalize
    data_med = np.nanmedian(data)
    data  /= data_med
    model /= data_med
    return 1.0/len(data) * np.nansum(np.abs(data-model))


def stddev(data, model):
    # Simple calculation of standard deviation statistic for a single fit
    return np.nanstd(data-model)


def med_abs_dev(data, model):
    # Simple calculation of median absolute deviation (MAD) statistic for a single fit
    return median_abs_deviation(data-model)


def ssr(data, model):
    # Simple calculation of the sum of squares of residuals (SSR) statistic for a single fit
    return np.sum((data-model)**2)


def ssr_test(resid_A, resid_B):
    '''
    Sum-of-Squares of Residuals test:
    The sum-of-squares of the residuals of the simple model (A)
    and the sum-of-squares of the residuals of complex model (B)
    '''

    ssr_resid_B = np.sum(resid_B**2)
    ssr_resid_A  = np.sum(resid_A**2)
    ssr_ratio = (ssr_resid_A)/(ssr_resid_B)
    return ssr_ratio, ssr_resid_A, ssr_resid_B


def anova_test(resid_A, resid_B, k_A, k_B):
    '''
    f-test:
    Perform an f-statistic for model comparison between two models.
    The f_oneway test is only accurate for normally-distributed values and should be compared
    against the Kruskal-Wallis test (non-normal distributions), as well as the Bartlett and Levene variance tests.
    We use the sum-of-squares of residuals for each model for the test.
    '''

    rss1 = np.sum(resid_A**2)
    rss2 = np.sum(resid_B**2)

    n = float(len(resid_B))
    dfn = k_B - k_A # deg. of freedom numerator
    dfd = n - k_B  # deg. of freedom denominator

    f_stat = ((rss1-rss2)/(k_B-k_A))/((rss2)/(n-k_B))
    f_pval = 1 - f.cdf(f_stat, dfn, dfd)
    conf = 1.0-(f_pval)
    return f_stat, f_pval, conf


def f_ratio(resid_A, resid_B):
    # The F-ratio is defined as the ratio in variances between two sets of data (residuals)
    return np.nanstd(resid_A)/np.nanstd(resid_B)


def chi2_metric(eval_ind, mccomps_A, mccomps_B):
    f_obs = mccomps_B['DATA'][0,:][eval_ind]/np.sum(mccomps_B['DATA'][0,:][eval_ind])
    f_exp = mccomps_B['MODEL'][0,:][eval_ind]/np.sum(mccomps_B['MODEL'][0,:][eval_ind])
    chi2_B, pval_B = chisquare(f_obs=f_obs,f_exp=f_exp)

    f_obs = mccomps_A['DATA'][0,:][eval_ind]/np.sum(mccomps_A['DATA'][0,:][eval_ind])
    f_exp = mccomps_A['MODEL'][0,:][eval_ind]/np.sum(mccomps_A['MODEL'][0,:][eval_ind])
    chi2_A, pval_A = chisquare(f_obs=f_obs,f_exp=f_exp)

    # The ratio of chi-squared values is defined as the improvement of model B over model A,
    chi2_ratio = 1.0-(chi2_B/chi2_A)

    return chi2_B, chi2_A, chi2_ratio


def normal_log_likelihood(data, model, sigma):
    # A simple normal log-likelihood for data, model, and noise
    return -0.5*np.sum((data-model)**2/sigma**2 + np.log(2*np.pi*sigma**2))


def calculate_BIC(mccomps_A, mccomps_B, k_A, k_B):
    # Calculates the Bayesian information criterion (BIC) for two models

    # Unpack the likelihood parameters
    data_A, model_A, noise_A = mccomps_A['DATA'], mccomps_A['MODEL'], mccomps_A['NOISE'] 
    data_B, model_B, noise_B = mccomps_B['DATA'], mccomps_B['MODEL'], mccomps_B['NOISE'] 
    ll_A = normal_log_likelihood(data_A, model_A, noise_A)
    ll_B = normal_log_likelihood(data_B, model_B, noise_B)

    bic_A = -2*ll_A+k_A*np.log(len(data_A))
    bic_B = -2*ll_B+k_B*np.log(len(data_B))
    bic_ratio = bic_B/bic_A

    return bic_A, bic_B, bic_ratio


def calculate_AIC(mccomps_A, mccomps_B, k_A, k_B):
    # Calculates the Akaike information criterion (AIC) for two models

    # Unpack the likelihood parameters
    data_A, model_A, noise_A = mccomps_A['DATA'], mccomps_A['MODEL'], mccomps_A['NOISE'] 
    data_B, model_B, noise_B = mccomps_B['DATA'], mccomps_B['MODEL'], mccomps_B['NOISE'] 
    ll_A = normal_log_likelihood(data_A, model_A, noise_A)
    ll_B = normal_log_likelihood(data_B, model_B, noise_B)

    aic_A = -2*ll_A+2*(k_A)
    aic_B = -2*ll_B+2*(k_B)
    aic_ratio = aic_B/aic_A

    return aic_A, aic_B, aic_ratio


def calculate_rsquared_ratio(mccomps_A, mccomps_B, eval_ind):
    data_A, model_A = mccomps_A['DATA'][0][eval_ind], mccomps_A['MODEL'][0][eval_ind]
    data_B, model_B = mccomps_B['DATA'][0][eval_ind], mccomps_B['MODEL'][0][eval_ind]

    # Since R-squared takes into account lines+continuum, we only want 
    # to be sensitive to flux that comes from lines, so we subtract
    # any contribution to the continuum from both before the calculation.
    # NOTE: this assumes that the continuum subtraction is generally good for both models
    cont_comps = ['HOST_GALAXY','POWER','APOLY','PPOLY','MPOLY','NA_OPT_FEII_TEMPLATE','BR_OPT_FEII_TEMPLATE',
                  'F_OPT_FEII_TEMPLATE','S_OPT_FEII_TEMPLATE','G_OPT_FEII_TEMPLATE','Z_OPT_FEII_TEMPLATE',
                  'UV_IRON_TEMPLATE','BALMER_CONT',]

    cont_model_A = np.zeros(len(data_A))
    cont_model_B = np.zeros(len(data_B))

    for comp in cont_comps:
        if comp in mccomps_A:
            comp_A = mccomps_A[comp][0][eval_ind]
            data_A - comp_A
            model_A - comp_A
        if comp in mccomps_B:
            comp_B = mccomps_B[comp][0][eval_ind]
            data_B - comp_B
            model_B - comp_B

    rsquared_A = 1 - (np.sum((data_A-model_A)**2))/(np.sum(data_A**2))
    rsquared_B = 1 - (np.sum((data_B-model_B)**2))/(np.sum(data_B**2))

    rsquared_ratio = rsquared_B/rsquared_A
    if not np.isfinite(rsquared_ratio):
        rsquared_ratio = 0.0

    return rsquared_A, rsquared_B, rsquared_ratio


def bayesian_AB_test(resid_A, resid_B, wave, noise, data, eval_ind, ddof, run_dir, plot=False):
    # Performs a Bayesian A/B hypothesis test for the likelihood distributions for two models

    # Smooth the noise using a 3-pixel Gaussian kernel
    noise = gaussian_filter1d(noise, 2.0, mode='nearest')

    # Sample the noise around the best-fit 
    nsamp = 10000
    resid_B_lnlike  = np.empty(nsamp)
    resid_A_lnlike = np.empty(nsamp)
    for i in range(nsamp):
        resid_B_lnlike[i] = np.sum(-0.5*(np.random.normal(loc=resid_B[eval_ind],scale=np.abs(noise[eval_ind]),size=len(eval_ind)))**2/noise[eval_ind]**2)
        resid_A_lnlike[i] = np.sum(-0.5*(np.random.normal(loc=resid_A[eval_ind],scale=np.abs(noise[eval_ind]),size=len(eval_ind)))**2/noise[eval_ind]**2)

    # Penalize by degrees of freedom
    resid_B_lnlike /= (len(data)-ddof)
    resid_A_lnlike /= (len(data))
    p_B = np.percentile(resid_B_lnlike, [16,50,84])
    p_A = np.percentile(resid_A_lnlike, [16,50,84])

    # The sampled log-likelihoods should be nearly Gaussian
    x = np.linspace(np.min([resid_B_lnlike, resid_A_lnlike]), np.max([resid_B_lnlike, resid_A_lnlike]), 1000)
    norm_B = norm(loc=p_B[1], scale=np.mean([p_B[2]-p_B[1], p_B[1]-p_B[0]]))
    norm_A = norm(loc=p_A[1], scale=np.mean([p_A[2]-p_A[1], p_A[1]-p_A[0]]))

    # Determine which distribution has the maximum likelihood.
    # Null Hypothesis, H0: B is no different than A
    # Alternative Hypothesis, H1: B is significantly different from A
    A = resid_A_lnlike
    A_mean = p_A[1]
    B = resid_B_lnlike
    ntrials = 10000
    B_samples = norm_B.rvs(size=ntrials)
    pvalues = np.array([(norm_A.sf(b)) for b in B_samples])*2.0
    pvalues[pvalues > 1] = 1
    pvalues[pvalues < 1e-6] = 0
    conf = (1 - pvalues)

    p_pval = np.percentile(pvalues, [16,50,84])
    p_conf = np.percentile(conf, [16,50,84])

    d = np.abs(p_B[1] - p_A[1]) # statistical distance
    disp = np.sqrt((np.mean([p_B[2]-p_B[1],p_B[1]-p_B[0]]))**2+(np.mean([p_A[2]-p_A[1],p_A[1]-p_A[0]]))**2) # total dispersion
    signif = d/disp # significance
    overlap = np.min([(p_B[2]-p_A[0]), (p_A[2]-p_B[0])]).clip(0) # 1-sigma overlap

    if plot:
        fontsize = 16
        fig = plt.figure(figsize=(18,10)) 
        gs = gridspec.GridSpec(2,4)
        gs.update(wspace=0.35, hspace=0.35)
        ax1 = plt.subplot(gs[0,0:4])
        ax2 = plt.subplot(gs[1,0])
        ax3 = plt.subplot(gs[1,1])
        ax4 = plt.subplot(gs[1,2])
        ax5 = plt.subplot(gs[1,3])

        plt.suptitle('BADASS A/B Likelihood Comparison Test', fontsize=fontsize)

        ax1.plot(wave[eval_ind], resid_A-resid_B, color='xkcd:bright red', linestyle='-', linewidth=1.0, label=r'$\Delta~\rm{Residuals}$')
        ax1.plot(wave[eval_ind], noise[eval_ind], color='xkcd:lime green', linestyle='-', linewidth=0.5,label='Noise')
        ax1.plot(wave[eval_ind], -noise[eval_ind], color='xkcd:lime green', linestyle='-', linewidth=0.5)
        ax1.axhline(0, color='xkcd:white', linestyle='--', linewidth=0.75)
        ax1.set_xlabel(r'$\lambda_{\rm{rest}}$ [$\rm{\AA}$]', fontsize=fontsize)
        ax1.set_ylabel(r'$f_\lambda$ [$10^{-17}$ erg cm$^{-2}$ s$^{-1}$ $\rm{\AA}^{-1}$]', fontsize=fontsize)
        ax1.set_title('Fitting Region Residuals', fontsize=fontsize)
        ax1.tick_params(axis='both', labelsize=fontsize)
        ax1.set_xlim(np.min(wave[eval_ind]), np.max(wave[eval_ind]))
        ax1.legend(fontsize=12)

        ax2.hist(resid_B_lnlike, bins='doane', histtype='step', label='Model B', density=True, color='xkcd:bright aqua', linewidth=0.5)
        ax2.axvline(p_B[1], color='xkcd:bright aqua', linestyle='--', linewidth=1)
        ax2.axvspan(p_B[0], p_B[2], alpha=0.25, color='xkcd:bright aqua')
        ax2.plot(x, norm_B.pdf(x), color='xkcd:bright aqua', linewidth=1)
        ax2.plot(x, norm_A.pdf(x), color='xkcd:bright orange', linewidth=1)

        ax2.hist(resid_A_lnlike, bins='doane', histtype='step', label='Model A', density=True, color='xkcd:bright orange', linewidth=0.5)
        ax2.axvline(p_A[1], color='xkcd:bright orange', linestyle='--', linewidth=1)
        ax2.axvspan(p_A[0], p_A[2], alpha=0.25, color='xkcd:bright orange')
        ax2.set_title('Log-Likelihood', fontsize=fontsize)
        ax2.tick_params(axis='both', labelsize=fontsize)
        ax2.legend()

        ax3.hist(pvalues, bins='doane', histtype='step', label='Model B', density=True, color='xkcd:bright aqua', linewidth=0.5)
        ax3.axvline(p_pval[1], color='xkcd:bright aqua', linestyle='--', linewidth=1)
        ax3.axvspan(p_pval[0], p_pval[2], alpha=0.25, color='xkcd:bright aqua')
        ax3.set_title(r'$p$-values', fontsize=fontsize)
        ax3.tick_params(axis='both', labelsize= fontsize)

        ax4.hist(conf, bins='doane', histtype='step', label='Model A', density=True, color='xkcd:bright aqua', linewidth=0.5)
        ax4.axvline(p_conf[1], color='xkcd:bright aqua', linestyle='--', linewidth=1)
        ax4.axvspan(p_conf[0], p_conf[2], alpha=0.25, color='xkcd:bright aqua')
        ax4.set_title(r'Confidence', fontsize=fontsize)
        ax4.tick_params(axis='both', labelsize= fontsize)

        ax5.axvline(0.0, color='black', label='$p$-value = %0.4f +/- (%0.4f, %0.4f)' % (p_pval[1],p_pval[2]-p_pval[1],p_pval[1]-p_pval[0]))
        ax5.axvline(0.0, color='black', label='Confidence = %0.4f +/- (%0.4f, %0.4f)' % (p_conf[1],p_conf[2]-p_conf[1],p_conf[1]-p_conf[0]))
        ax5.axvline(0.0, color='black', label='Statistical Distance = %0.4f' % d)
        ax5.axvline(0.0, color='black', label='Combined Dispersion  = %0.4f' % disp)
        ax5.axvline(0.0, color='black', label=r'Significance ($\sigma$) = %0.4f' % signif)
        ax5.axvline(0.0, color='black', label=r'$1\sigma$ Overlap = %0.4f' % overlap)
        ax5.legend(loc='center', fontsize=fontsize, frameon=False)
        ax5.axis('off')
        
        fig.tight_layout()
        plt.savefig(run_dir.joinpath('test_results.pdf'))
        plt.close()

    return p_pval[1],p_pval[2]-p_pval[1],p_pval[1]-p_pval[0], p_conf[1],p_conf[2]-p_conf[1],p_conf[1]-p_conf[0], d, disp, signif, overlap


def calculate_aon(test, line_list, mccomps, noise):
    # Calculates the amplitude-over-noise for the maximum of all lines being tested for a given test

    full_profile = np.zeros(len(mccomps['WAVE'][0]))
    for l in line_list:
        if (l in test) or (('parent' in line_list[l]) and (line_list[l]['parent'] in test)):
            full_profile += mccomps[l][0]

    avg_noise = np.nanmean(noise)
    aon = np.nanmax(full_profile)/avg_noise
    return aon



def collect_test_metrics(ctx, fit_results_A, fit_results_B, line_name):
    fit_mask = ctx.target.fit_mask
    mccomps_A = fit_results_A['mccomps']
    resid_A = mccomps_A['RESID'][0][fit_mask]
    mccomps_B = fit_results_B['mccomps']
    resid_B = mccomps_B['RESID'][0][fit_mask]

    metrics = {}
    ddof = np.abs(fit_results_A['dof']-fit_results_B['dof'])
    _,_,_,conf,_,_,_,_,_,_ = bayesian_AB_test(resid_A, resid_B, ctx.fit_wave[fit_mask], ctx.fit_noise[fit_mask], ctx.fit_spec[fit_mask], np.arange(len(resid_A)), ddof, ctx.target.options.io_options.output_dir, plot=False)
    metrics['BADASS'] = conf

    ssr_ratio, ssr_A, ssr_B = ssr_test(resid_A, resid_B)
    metrics['SSR_RATIO'] = ssr_ratio

    k_A, k_B = fit_results_A['npar'], fit_results_B['npar']
    f_stat, f_pval, f_conf = anova_test(resid_A, resid_B, k_A, k_B)
    metrics['ANOVA'] = f_conf

    aic_A, aic_B, aic = calculate_AIC(mccomps_A, mccomps_B, k_A, k_B)
    metrics['AIC'] = aic

    bic_A, bic_B, bic = calculate_BIC(mccomps_A, mccomps_B, k_A, k_B)
    metrics['BIC'] = bic

    metrics['F_RATIO'] = f_ratio(resid_A, resid_B)

    chi2_B, chi2_A, chi2_ratio = chi2_metric(np.arange(len(resid_A)), mccomps_A, mccomps_B)
    metrics['CHI2_RATIO'] = chi2_ratio

    rsquared_A, rsquared_B, rsquared_ratio = calculate_rsquared_ratio(mccomps_A, mccomps_B, check_ind)
    metrics['RCHI2_RATIO'] = rsquared_ratio

    return metrics



def thresholds_met(test_options, cur_metrics, fit_results):
    pass_list = [cur_metrics[metric] >= thresh for metric, thresh in test_options.metrics.items() if metric in cur_metrics]
    if 'AON' in test_options.metrics: pass_list.append(fit_results['aon'] >= test_options.metrics['AON']) # special case
    mode_func = {'any':np.any, 'all':np.all}[test_options.conv_mode]
    return mode_func(pass_list)
