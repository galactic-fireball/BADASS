import numpy as np

def pca_reconstruction(target):
    '''
    Performs principal component analysis (PCA) on input spectrum using SDSS template spectra to reconstruct specific, user-specified, spectral regions.
    Input:
        wave_input - The input spectrum wavelength array
        flux_input - The input spectrum flux array
        err_input - The input spectrum flux error array
        n_components - Int or None. If int, chooses how many principal components to calculate and return. If None, calculates all available components. Default is 20.
        masks - List of tuples that define regions over which PCA should be performed. Default is 4000-4500 A. 
        plot_pca - Boolean of plot PCA or not. If True, returns PCA spectrum overplotting original spectrum, with residuals shown below.  
        
    Output:
        new_flux - New flux array, with PCA reconstruction of corresponding flux values
        flux_resid - Residual flux array of input spectrum and PCA reconstructed spectrum 
        err_flux - Final flux error (either 0.1*flux or the original error (if original error has no nans) )
        evecs - Eigenvectors from PCA
        evals_cs - Cummulative sum of normalized eigenvalues, i-th component tells us percentage of explained variance using i eigenspectra. evals_csv[-1] is explained variance of final component
        spec_mean - Array of mean values of eigenspectra
        coeff - Coefficients used in reconstruction of spectrum
    '''

    # Regardless of PCA, check for nans in flux and flux error arrays. If found, raise an error because they will prevent fit optimization
    if not target.options.pca_options.do_pca:
        if (np.isnan(target.spec).any()) or (np.isnan(target.noise).any()):
            raise ValueError("The flux or flux error in fitting region ({mi}, {ma}) is nan, stopping fit. Change fitting region or enable PCA to cover nan region.".format(mi=target.fit_reg.min, ma=target.fit_reg.max))
        target.log.pca_information()
        return

    pca_masks = target.options.pca_options.pca_masks
    target.log.info(f"Performing PCA on a spectrum with nans over region(s) {pca_masks}. Be careful to ensure PCA covers all nan regions, else PCA will fail.")
    target.log.info(" Performing PCA analysis...\n")
    if len(pca_masks):
        pca_reg_test = [(i[0]>=target.fit_reg.min,i[1]<=target.fit_reg.max) for i in pca_masks] # check that pca mask regions are within fitting region
        if not np.all(pca_reg_test):
            raise ValueError("PCA region masks {pca_masks} must be within fitting region ({mi}, {ma})".format(pca_masks=pca_masks, mi=target.fit_reg.min, ma=target.fit_reg.max))
    else:
        pca_masks = ([(target.fit_reg.min, target.fit_reg.max)])
        target.options.pca_options.pca_masks = pca_masks

    wave_input = np.array(target.wave)
    flux_input = np.array(target.spec)
    err_input = np.array(target.noise)
    flux_mean = np.nanmean(flux_input)
    # download reconstructed SDSS spectra to be used as templates 
    data = sdss_corrected_spectra.fetch_sdss_corrected_spectra()

    spectra_corr = sdss_corrected_spectra.reconstruct_spectra(data) # "eigenspectra"
    wavelengths = sdss_corrected_spectra.compute_wavelengths(data)
    spectra_corr_interp = []
    
    flux_nan_check = np.isnan(flux_input).any()
    err_nan_check = np.isnan(err_input).any()
    
    if flux_nan_check:
        target.log.debug('\tnans detected in spectrum flux. Setting to spectrum mean and performing PCA.')
        flux_nan, flux_nan_func = nan_helper(flux_input)
        flux_nan_ind = flux_nan_func(flux_nan)
        for fni in flux_nan_ind:
            mask_check = []
            flux_wave_nan = wave_input[fni]
            for m in pca_masks:
                chk = ( (m[0] <= flux_wave_nan) and  ( m[1] >= flux_wave_nan) )
                mask_check.append(chk)

            if not np.any(mask_check):
                raise ValueError(f"Wavelength {flux_wave_nan} has a nan flux, but is not covered by PCA. Adjust your PCA region.")

        flux_input[flux_nan_ind] = flux_mean*np.ones(len(flux_nan_ind))

    if err_nan_check:
        target.log.debug('\tnans detected in spectrum flux error. Setting to 0.1*flux at corresponding wavelength.')
        err_nan, err_nan_func = nan_helper(err_input)
        err_nan_ind = err_nan_func(err_nan)

        for eni in err_nan_ind:
            mask_check = []
            err_wave_nan = wave_input[eni]
            for m in pca_masks:
                chk = ( (m[0] <= err_wave_nan) and  ( m[1] >= err_wave_nan) )
                mask_check.append(chk)

            if not np.any(mask_check):
                raise ValueError(f"Wavelength {err_wave_nan} has a nan flux err, but is not covered by PCA. Adjust your PCA region.")

        if not flux_nan_check:
            err_input[err_nan_ind] = np.abs(0.1* flux_input[err_wave_nan])  #flux_mean*np.ones(len(err_nan_ind)) # if only errors have nans, then correct right away


    # interpolate reconstructed SDSS spectra to match input spectrum dimension. Assumes template dimension is less than input dimension
    for spec in spectra_corr:
        s_interp = np.interp(wave_input,wavelengths,spec)
        spectra_corr_interp.append(s_interp)

    spectra_corr_interp = np.array(spectra_corr_interp) # need to convert to numpy array for consistency

    # fit spectrum for eigenvalues
    n_components = target.options.pca_options.n_components
    if isinstance(n_components, int):
        pca = PCA(n_components = n_components) # optional n_components = 4,5,... argument here
    elif isinstance(n_components, type(None)):
        pca = PCA()
    else:
        target.log.warning(f"\tWarning: {n_components} is invalid argument for number of PCA components. Must be int or None. Defaulting to 20 components.")
        pca = PCA(n_components = 20)

    # gather relevant output results
    pca.fit(spectra_corr_interp)
    evals = pca.explained_variance_ratio_ # eigenvalue ratio -- tells us PERCENTAGE of explained variance. NOT ACTUAL EIGENVALUES. Use explained_variance_ to get eigenvalues of covariance matrix
    evals_cs = evals.cumsum()
    evecs = pca.components_ # corresponding eigenvectors

    # calculate template spectra means
    spec_mean = spectra_corr_interp.mean(0)

    coeff = np.dot(evecs, flux_input-spec_mean) # project CENTERED input spectrum onto eigenspectra
    final_flux = spec_mean + np.dot(coeff, evecs) # flux arr of reconstructed spectrum using all computed components 

    # replace original flux with new flux, but only for masked region(s)
    new_flux = np.array(flux_input) # if not array, will break

    err_flux = np.array(err_input) # initialize flux error array
    # for each mask, replace original flux values with PCA flux values
    for mask in pca_masks:
        ind_low = find_nearest(wave_input, mask[0])[1]
        ind_upp = find_nearest(wave_input, mask[1])[1]

        new_flux_vals = final_flux[ind_low:ind_upp]
        new_flux[ind_low:ind_upp] = new_flux_vals

    if err_nan_check and flux_nan_check:
        err_flux[err_nan_ind] = np.abs(0.1 * new_flux[err_nan_ind]) # if both flux and errors have nans, then replace error nans with errors from pca flux 

    flux_resid = flux_input-new_flux
    if target.options.plot_options.plot_pca:
        plt.style.use('default') # lazy way of switching plot styles to default (and back)
        fig,ax = plt.subplots(nrows = 2, ncols = 1, sharex = False, sharey = False, figsize = (18,10))
        fig.suptitle('PCA Reconstruction',size = 22)
        ax0 = ax[0]
        ax1 = ax[1]
        ax0.plot(wave_input, flux_input, label = 'Input Spectrum', color = 'dimgray')
        ax0.plot(wave_input, new_flux, label = 'PCA Spectrum', color = 'k')
        ax1.plot(wave_input, flux_resid, color = 'k')

        #ax0.set_ylabel('Flux', size = 16)
        ax0.set_ylabel(r'$f_\lambda$ ($10^{-17}$ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$)', size = 16)
        ax1.set_ylabel(r'$f_\lambda$ Residual', size = 16)

        for i,mask in enumerate(pca_masks):
            ax0.axvspan(mask[0], mask[1], color = 'lightgray', label = 'PCA Region(s)' if i == 0 else "", alpha = 0.5)
        ax0.legend()

        if n_components == 0:
                text = "mean + 0 components"
        elif n_components == 1:
            text = "mean + 1 component\n"
            text += r"$(\sigma^2_{{tot}} = {0:.4f})$".format(evals_cs[-1])
        elif n_components is None:
            text = "mean + all components\n"
            text += r"$(\sigma^2_{{tot}} = {0:.4f})$".format(evals_cs[-1])
        else:
            text = f"mean + {n_components} components\n"
            text += r"$(\sigma^2_{{tot}} = {0:.4f})$".format(evals_cs[-1])

        ax1.text(0.01, 0.97, text, ha='left', va='top', transform=ax1.transAxes, bbox = dict(facecolor='none', edgecolor='black',boxstyle='round,pad=0.5'))
        plt.xlabel(r'${\rm Wavelength\ (\AA)}$', size = 16)
        plt.tight_layout()
        plt.savefig(target.outdir.joinpath('pca_spectrum.pdf'))
        plt.style.use('dark_background')
        plt.close(fig)

    target.log.info(" PCA analysis complete!")
    target.log.info("---------------------------------------")
    target.log.pca_information(pca_nan_fix=True, pca_exp_var=evals_cs[-1])

    # Use new flux and error arrays
    target.spec = new_flux
    target.noise = err_flux

    return new_flux, flux_resid, err_flux, evecs, evals_cs, spec_mean, coeff




