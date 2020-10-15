# -*- coding: utf-8 -*-
"""
An extension for the MNE-Python repository (Gramfort et al., 2013) implementing the FASTER method (Nolan, Whelan, & Reilly, 2010)
References:
-----------
    Gramfort, A., Luessi, M., Larson, E., Engemann, D., Strohmeier, D., Brodbeck, C., Goj, R., Jas, M., Brooks, T., Parkkonen, L., & Hämäläinen, M. (2013). MEG and EEG data analysis with MNE-Python. Frontiers in Neuroscience, 7.
    Nolan, H., Whelan, R., & Reilly, R. (2010). FASTER: Fully Automated Statistical Thresholding for EEG artifact Rejection. Journal Of Neuroscience Methods, 192(1), 152-162.    
"""

# In[]
### Importing module dependencies ###

import os
import mne
import numpy as np
import pandas as pd
from scipy import stats, signal
from multiprocessing import Pool, cpu_count


# In[]
### Defining number of workers for multiprocessing

n_free = 0
workers = cpu_count() - n_free

# In[]
### Hurst exponent function from 'eegfaster' ###

def _hurst(x):
    """
    FASTER implementation of the Hurst Exponent.
    
    Parameters
    ----------
    x:  numpy.ndarray
        Vector with the data sequence.
    
    Returns
    -------
    h:  float
        Computed Hurst-exponent.
    """

    # Get a copy of the data
    x0 = x.copy()
    x0_len = len(x)

    yvals = np.zeros(x0_len)
    xvals = np.zeros(x0_len)
    x1 = np.zeros(x0_len)

    index = 0
    binsize = 1

    while x0_len > 4:

        y = x0.std()
        index += 1
        xvals[index] = binsize
        yvals[index] = binsize*y

        x0_len /= 2
        binsize *= 2
        for ipoints in xrange(x0_len):
            x1[ipoints] = (x0[2*ipoints] + x0[2*ipoints - 1])*0.5

        x0 = x1[:x0_len]

    # First value is always 0
    xvals = xvals[1:index+1]
    yvals = yvals[1:index+1]

    logx = np.log(xvals)
    logy = np.log(yvals)

    p2 = np.polyfit(logx, logy, 1)
    
    return p2[0]
    


# In[]
### Function to compute correct polar angle distances from the reference electrode ###

def _compute_correct_thetas(mont, ref_ch, eeg_chs):
    """
    Compute correct angular distances from the reference electrode.
    
    Note: Current version works correctly only for reference electrodes on the median-sagittal plane!
    """
    # Getting default channel positions from montage
    data_mont = pd.DataFrame(mont.pos, index=mont.ch_names, columns=['x', 'y', 'z'])
    
    # Getting the current polar angle of the reference electrode
    theta_ref = np.arctan2(np.sqrt(data_mont.loc[ref_ch]['x']**2 + data_mont.loc[ref_ch]['y']**2), data_mont.loc[ref_ch]['z'])
    
    # Rotating the sensor-space to the correct position
    theta_rot = (360 * np.pi / 180) - theta_ref
    matrix_rot = np.array([[1,0,0], [0, np.cos(theta_rot), -np.sin(theta_rot)], [0, np.sin(theta_rot), np.cos(theta_rot)]])
    
    rotated = np.empty((0,3))
    for ch in data_mont.iterrows():
        rotated = np.vstack((rotated, np.array(np.dot(np.asarray(ch[1]), matrix_rot))))
    rotated = pd.DataFrame(rotated, index=mont.ch_names, columns=['x', 'y', 'z'])
    
    # Getting the corrected polar angles
    thetas = [np.arctan2(np.sqrt(ch[1]['x']**2 + ch[1]['y']**2), ch[1]['z']) for ch in rotated.iterrows() if ch[0] in eeg_chs]
    
    return thetas
    
    
 
# In[]
### Functions for single parameters (multiprocessing.Pool.map() needs top level functions which are pickleable). ###
    
def _compute_channels_corr(ch_pair):
    """Compute Pearson's correlation coefficient between two channels (absolute)"""
#    chs_corr = abs(stats.pearsonr(data[ch_pair[0]], data[ch_pair[1]])[0])
    chs_corr = abs(np.corrcoef(data[ch_pair[0]], data[ch_pair[1]]))
    return (ch_pair, chs_corr)


def _compute_channel_mcorr(ch):
    """Compute the mean correlation coefficient describing a given channel"""
    ch_mcorr = np.mean([corr_dict[ch_pair] for ch_pair in corr_dict.keys() if ch in ch_pair])
    return ch_mcorr


def _compute_channel_var(ch):
    """Compute the variance describing a given channel"""
    ch_var = np.var(data[ch])
    return ch_var


def _compute_channel_hurst(ch):
    """Compute the Hurst exponent describing a given channel"""
    ch_hurst = _hurst(np.asarray(data[ch]))
    return ch_hurst

    
def _compute_channel_ampl(ch):
    """Compute the amplitude describing a given channel"""
    ch_ampl = max(data[ch]) - min(data[ch])
    return ch_ampl


def _compute_channel_dev(ch):
    """Compute the deviance describing a given channel"""
    ch_dev = abs(means[ch] - np.mean(data[ch]))
    return ch_dev.values[0]  


def _compute_channel_median_grad(ch):
    """Compute the median gradient describing a given channel"""
    ch_med_grad = np.median(np.diff(data[ch])) 
    return ch_med_grad


def _compute_component_eog_corr(comp):
    """Compute the maximum correlation coefficient with EOG channels describing a given ICA component"""
    c_max_eog = max([abs(stats.pearsonr(data[comp], eog_data[ch])[0]) for ch in eog_chs])
    return c_max_eog


def _compute_component_spatial_kurt(comp):
    """Compute the spatial kurtosis describing a given ICA component"""
    c_kurt = stats.kurtosis(maps[comp])
    return c_kurt


def _compute_component_mean_slope(comp):
    """Compute the mean slope in the frequency band describing a given ICA component"""
    freqs, spectra = signal.welch(data[comp], window='hamming', fs=samp_freq)
    c_mean_slope = np.mean(np.diff(10*np.log10([spectra[i] for i, f in enumerate(freqs) if f > band[0] and f < band[1]])))
    return c_mean_slope


def _compute_component_hurst(comp):
    """Compute the Hurst exponent describing a given ICA component"""
    c_hurst = _hurst(np.asarray(data[comp]))
    return c_hurst


def _compute_component_median_grad(comp):
    """Compute the median gradient describing a given ICA component"""
    c_med_grad = np.median(np.diff(data[comp])) 
    return c_med_grad


# In[]    
### Initializer for Pool objects
    
#def init_pool(var1=None, var2=None, var3=None):
#    global data, corr_dict
#    data = var1
#    corr_dict = var2
#    means = var3


# In[] 
##### The actual stages of FASTER #####    
    
### Finding bad channels in raw data (Stage I.) ###

def find_bad_channels(raw, mont, params=['mean_corr', 'var','hurst'], crit_z=3.0, ref_ch='E11', apply_ref=True, f_band=(0.1, 95), f_band_fs=(2, 95), apply_band=True, f_notch=50, apply_notch=True, interpolate=True, plot=False):
    """
    Implements the first stage of FASTER: finds the channel outliers in the raw data.
    
    Parameters
    ----------
    raw:        mne.io.Raw
                The raw data.
    mont:       mne.channels.Montage
                The montage containing the electrode locations.
    params:     list
                The parameters to use. Default is to use them all.
                    'mean_corr':    The corrected mean correlation coefficient describing a given channel (Parameter 1.)
                    'var':          The corrected variance describing a given channel (Parameter 2.)
                    'hurst':        The corrected Hurst-exponent describing a given channel (Parameter 3.)
    crit_z:     float
                The critical z-score to use when determining outliers. Defaults to 3.0.
    ref_ch:     string
                The label of the reference electrode. Defaults to 'E11' (the Fz equivalent electrode of the GSN-Hydrocel-128 caps).
    apply_ref:  boolean
                Whether to apply custom reference using 'ref_ch'. Defaults to True.
    f_band:     tuple
                The low-pass and high-pass values [Hz] to use when filtering the raw data. Defaults to (4,30).
    apply_filt: boolean
                Whether to apply a bandpass filter using 'f_band'). Defaults to True.
    interpolate:boolean
                Whether to modify 'raw' by interpolating the channels marked as outliers. Defaults to True.
    recursive:  boolean
                Whether to use recursive outliering method. Defaults to False.
    
    Returns
    -------
    raw:    mne.io.Raw
            The raw data (modified or unmodified).
    bads:   list
            The labels of the channels marked as outliers.
    """
    # Definig global variables for multiprocessing
    global data, corr_dict
    
    # Getting requested parameters
    funcs = []
    if 'mean_corr' in params:
        funcs.append(_compute_channel_mcorr)
    if 'var' in params:
        funcs.append(_compute_channel_var)
    if 'hurst' in params:
        funcs.append(_compute_channel_hurst)
    
	# Making a copy for FASTER to work with
    raw_fs = raw.copy()
    
    # Preloading data
    raw.load_data()
    raw_fs.load_data()
    
    # Setting the Fz electrode as reference (if chosen)    
    if apply_ref:
        raw, _ = mne.io.set_eeg_reference(raw, [ref_ch])
        raw_fs, _ = mne.io.set_eeg_reference(raw_fs, [ref_ch])
    
    # Applying band-pass and notch filters (if chosen)
    if apply_band:
        #raw.filter(f_band[0], f_band[1], method='fir')
        #raw_fs.filter(f_band_fs[0], f_band_fs[1], method='fir')
        raw.filter(f_band[0], f_band[1], method='iir', iir_params=dict(order=5, ftype='butter', output='sos')) # Steeper filter
        raw_fs.filter(f_band_fs[0], f_band_fs[1], method='iir', iir_params=dict(order=5, ftype='butter', output='sos')) 
    if apply_notch:
        raw.notch_filter(freqs=f_notch, method='fir')
        raw_fs.notch_filter(freqs=f_notch, method='fir')
        #raw.notch_filter(freqs=f_notch, method='iir', iir_params=dict(order=5, ftype='butter', output='sos'))
        #raw_fs.notch_filter(freqs=f_notch, method='iir', iir_params=dict(order=5, ftype='butter', output='sos'))
        
    # Getting data from the mne.io.Raw object
    data = raw_fs.to_data_frame()
    # Getting the names of EEG channels
    eeg_chs = [raw_fs.info['ch_names'][i] for i in mne.pick_types(raw.info, meg=False, eeg=True) if raw_fs.info['ch_names'][i] != ref_ch]
    
    # Getting channel correlations for Parameter 1.
    if 'mean_corr' in params:
        # Getting all possible channel pairings
        ch_pairs = []
        for ch_x in eeg_chs:
            ch_pairs += [(ch_x, ch_y) for ch_y in eeg_chs if ch_x != ch_y and (ch_y, ch_x) not in ch_pairs]
        #Getting correlation coefficients
        pool = Pool(workers)
        corr_dict = dict(pool.map(_compute_channels_corr, ch_pairs))
        pool.close()
        pool.join()
    
    # Getting correct polar angles from the reference electrode
    thetas = _compute_correct_thetas(mont, ref_ch, eeg_chs)
    
    # Defining channel outliers
    pool = Pool(workers)
    bads = []
    for func in funcs:
        values = pool.map(func, eeg_chs)
        # Quadratic correction for distance from reference electrode
        p = np.polyfit(thetas, values, 2)
        fitcurve = np.polyval(p, thetas)
        values = list(values - fitcurve)
        # Outliering based on critical z-score
        chs = eeg_chs[:]
        z_chs = stats.zscore(values)
        bads += [chs[i_ch] for i_ch, z_ch in enumerate(z_chs) if abs(z_ch) > crit_z and chs[i_ch] not in bads]	
    pool.close()
    pool.join()
    
    # Deleting global variables
    del data, corr_dict
    
    # Plotting bad channels
    print 'Marked %i channel outlier(s) for interpolation (FASTER):' %len(set(bads))
    print bads
    raw.info['bads'] = set(bads)
    raw_fs.info['bads'] = set(bads)
    if plot:
        raw.plot(bad_color='red')
    
    # Interpolating bad channels (if chosen)
    if interpolate:
        raw.interpolate_bads(reset_bads=True)
        raw_fs.interpolate_bads(reset_bads=True)
    
    return raw, raw_fs, bads

    
# In[]
### Finding bad epochs (Stage II.) ###

def find_bad_epochs(raw, raw_fs, events, event_id, params=['ampl','dev','var'], crit_z=3.0, tmin=-0.5, tmax=1.5, baseline=(-0.2,0), decimate=False, dc=10, drop=True, drop_eog_events=False, eog_tresh=200, drop_errors=False, errors=None):
    """
    Implements the second step of FASTER: finds the epoch outliers in the epoched data.
    
    Parameters
    ----------
    raw:        mne.io.Raw
                The raw data. Preferably returned by mnefaster.find_bad_channels().
    events:     numpy.array
                The events to use when epoching the raw data. Preferably returned by mne.find_events().
    event_id:   list
                The types of events to use when epoching the raw data.
    params:     list
                The parameters to use. Default is to use them all.
                    'ampl': The mean of maximum channel amplitudes describing a given epoch (Parameter 3.)
                    'dev':  The mean of absolute channel deviations describing a given epoch (Parameter 4.)
                    'var':  The mean of channel variances describing a given epoch (Parameter 5.)
    crit_z:     float
                The critical z-score to use when determining outliers. Defaults to 3.0.
    t_min:      float
                The starting point of the epochs relative to the events. Defaults to -0.5.
    t_max:      float
                The end point of the epochs relative to the events. Defaults to 1.5.
    baseline:   tuple
                The low-pass and high-pass values [Hz] to use when filtering the raw data. Defaults to (4,30).
    drop:       boolean
                Whether to modify 'epochs' by excluding the epochs marked as outliers. Defaults to True.
    recursive:  boolean
                Whether to use recursive outliering method. Defaults to False.
                
    Returns
    -------
    epochs: mne.io.Epochs
            The epoched data (modified or unmodified).
    bads:   list
            The indices of the epochs marked as outliers.
    """
    # Definig global variables for multiprocessing
    global data, means
    
    # Getting requested parameters
    funcs = []
    if 'ampl' in params:
        funcs.append(_compute_channel_ampl)
    if 'dev' in params:
        funcs.append(_compute_channel_dev)
    if 'var' in params:
        funcs.append(_compute_channel_var)
    
    # Getting the names of channels
    eeg_chs = [raw.info['ch_names'][i] for i in mne.pick_types(raw.info, meg=False, eeg=True, eog=False)]
    eog_chs = [raw.info['ch_names'][i] for i in mne.pick_types(raw.info, meg=False, eeg=False, eog=True)]
    
    # Creating Epochs object from Raw
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=baseline, picks=mne.pick_types(raw.info, meg=False, eeg=True, eog=True)).drop_bad()
    epochs_fs = mne.Epochs(raw_fs, events, event_id, tmin, tmax, baseline=baseline, picks=mne.pick_types(raw.info, meg=False, eeg=True, eog=True)).drop_bad()
    
    # Preloading and resampling (using decimation according to the recommendations of MNE)
    epochs.load_data()
    epochs_fs.load_data()
    
    if decimate:
        epochs.decimate(dc)
        epochs_fs.decimate(dc)
        
    # Drop erroneous trials
    if drop_errors:
        epochs.drop(errors)
        epochs_fs.drop(errors)
        epochs.apply_baseline(baseline)
        epochs_fs.apply_baseline(baseline)
          
    # Drop EOG events based on treshold
    if drop_eog_events:
        epochs_data = epochs_fs.to_data_frame()
        eog_bads = []
        for i_epoch in set(epochs_data.index.get_level_values(level='epoch')):
            data = epochs_data.xs(i_epoch, level='epoch')
            eog_maxes = [max(data[ch]) - min(data[ch]) for ch in eog_chs]
            if max(eog_maxes) > eog_tresh:    
                eog_bads.append(i_epoch)
        if len(eog_bads) > i_epoch-19:
            del eog_bads[:20] # So that Epochs object doesn't return empty
        epochs.drop(eog_bads)
        epochs_fs.drop(eog_bads)
        epochs.apply_baseline(baseline)
        epochs_fs.apply_baseline(baseline)
        
    # Getting epochs data
    epochs_data = epochs_fs.to_data_frame()
    
    # Getting channel means across epochs for Parameter 5.
    means = pd.DataFrame([np.mean(epochs_data[ch]) for ch in eeg_chs], index=eeg_chs).transpose()
    
    # Iterating over epochs
    epochs_values = np.empty((0,3))
    for i_ep in set(epochs_data.index.get_level_values(level='epoch')):
        # Getting data of single epoch
        data = epochs_data.xs(i_ep, level='epoch')
        # Computing all parameters for current epoch
        pool = Pool(workers)
        values = []
        for func in funcs:        
            values.append(np.mean(pool.map(func, eeg_chs)))
        epochs_values = np.vstack([epochs_values, np.asarray([values])])
        pool.close()
        pool.join()
        
    # Deleting global variables
    del data, means
    
    # Outliering based on critical z-score
    epochs_z = stats.zscore(epochs_values)
    bads = [i_ep for i_ep, z_ep in enumerate(epochs_z) if any(abs(z) > crit_z for z in z_ep)]
    
    # Plotting bad epochs
    print 'Marked %i epoch outlier(s) for exclusion (FASTER):'%len(bads)
    print bads    
    
    # Dropping epoch outliers (if chosen)
    if drop:
        epochs.drop(bads)
        epochs_fs.drop(bads)
    
    return epochs, epochs_fs, bads
    

# In[]
### Fitting ICA with predefined parameters ###

def fit_ica(epochs, epochs_fs, n_interpolated, apply_av=True, ica_method='picard', baseline=(-0.2, 0)):
    """
    Implements the ICA process used by FASTER.
    
    Parameters
    ----------
    epochs:         mne.io.Epochs
                    The epoched data. Preferably returned by mnefaster.find_bad_epochs().
    n_interpolated: int
                    The number of interpolated channels in the data.
    apply_ref:      boolean
                    Whether to apply an average reference projection. Defaults to True.
    method:         string
                    ICA method to use. Passed to mne.preprocessing.ICA.
    
    Returns
    -------
    epochs: mne.io.Epochs
            The epoched data (modified or unmodified).
    ica:    mne.preprocessing.ICA
            The ICA applied on 'epochs'.
    """
    # Getting the names of channels
    eeg_chs = [epochs.info['ch_names'][i] for i in mne.pick_types(epochs.info, meg=False, eeg=True)]
    eog_chs = [epochs.info['ch_names'][i] for i in mne.pick_types(epochs.info, meg=False, eog=True)]
    
    # Setting average as reference if epochs is still Fz-referenced (if chosen)
    if apply_av:
        epochs.load_data()
        epochs, _ = mne.io.set_eeg_reference(epochs, eeg_chs)
        epochs.apply_baseline(baseline)
        
        epochs_fs.load_data()
        epochs_fs, _ = mne.io.set_eeg_reference(epochs_fs, eeg_chs)
        epochs_fs.apply_baseline(baseline)
        
    # Temporarily setting eog channels as EEG to include them in the ICA   
    for ch in eog_chs: 
        epochs_fs.set_channel_types({ch:'eeg'}) 
        
    # Defining the correct number of PCA components   
    c_pca = int(min(np.floor(np.sqrt(len(epochs_fs.to_data_frame().index)/25)), epochs_fs.info['nchan'] - n_interpolated))
    
    # Running ICA with the chosen method
    ica = mne.preprocessing.ICA(method=ica_method, n_components=c_pca-1, max_pca_components=c_pca, max_iter=200)
    ica.fit(epochs_fs)
    
    # Resetting the eog channels
    for ch in eog_chs: 
        epochs_fs.set_channel_types({ch:'eog'})
        
    return epochs, epochs_fs, ica


# In[]
### Finding bad components (Stage III.) ###
    
def find_bad_components(epochs, epochs_fs, ica, params=['eog_corr','spatial_kurt','mean_slope','hurst','median_grad'], crit_z=3.0, subtract=True, baseline=(-0.2, 0), plot=False):
    """
    Implements the third stage of FASTER: finds the component outliers of the ICA fitted on the epoched data.
    
    Parameters
    ----------
    epochs:     mne.io.Epochs
                The epoched data returned by mnefaster.fit_ica().
    ica:        mne.preprocessing.ICA
                The ICA object returned by mnefaster.fit_ica().
    params:     list
                The parameters to use. Default is to use them all.
                    'eog_corr':     The maximum correlation coefficient with eog channels describing a given component (Parameter 6.)
                    'spatial_kurt': The kurtosis of the spatial information in a given component (Parameter 7.)
                    'mean_slope':   The mean slope in the frequency band describing a given component (Parameter 8.)
                    'hurst':        The Hurst-exponent describing a given component (Parameter 9.)
                    'median_grad':  The median gradient describing a given component (Parameter 10.)
    crit_z:     float
                The critical z-score to use when determining outliers. Defaults to 3.0.
    subtract:   boolean
                Whether to subtract components marked as outliers from epoched data. Defaults to True.
    recursive:  boolean
                Whether to use recursive outliering method. Defaults to False.
    
    Returns
    -------
    epochs: mne.io.Epochs
            The epoched data (modified or unmodified).
    bads:   list
            The indices of ICA components marked as outliers.
    """
    # Definig global variables for multiprocessing
    global data, eog_chs, eog_data, maps, band, samp_freq 
    
    # Getting requested parameters
    funcs = []
    if 'eog_corr' in params:
        funcs.append(_compute_component_eog_corr)
    if 'spatial_kurt' in params:
        funcs.append(_compute_component_spatial_kurt)
    if 'mean_slope' in params:
        funcs.append(_compute_component_mean_slope)
    if 'hurst' in params:
        funcs.append(_compute_component_hurst)
    if 'median_grad' in params:
        funcs.append(_compute_component_median_grad)

    # Getting the names of eog channels
    eog_chs = [epochs_fs.info['ch_names'][i] for i in mne.pick_types(epochs_fs.info, meg=False, eog=True)]
    
    # Getting temporal data of ICA components
    data = ica.get_sources(epochs_fs).to_data_frame()
    
    # Getting spatial data of ICA components (jona-sassenhagen)
    maps = pd.DataFrame(np.dot(ica.mixing_matrix_[:, list(range(ica.n_components_))].T, ica.pca_components_[:ica.n_components_]), index=data.columns.values).transpose()

    # Getting epochs data (for EOG correlations)
    eog_data = epochs_fs.to_data_frame()
    
    # Getting high- and lowpass values
    band = (epochs_fs.info['highpass'], epochs_fs.info['lowpass'])
    
    # Getting sampling frequency
    samp_freq = epochs_fs.info['sfreq']
    
    # Defining component outliers based on critical z-score
    pool = Pool(workers)
    
    bads = []
    names = list(data.columns.values)
    for func in funcs:
        values_comps = list(pool.map(func, data.columns.values))
        z_comps = stats.zscore(values_comps)
        bads_i = [i_comp for i_comp, z_comp in enumerate(z_comps) if abs(z_comp) > crit_z]
        bads += [names[bad] for bad in bads_i if names[bad] not in bads]
        
    pool.close()
    pool.join()
    
    # Deleting global variables
    del data, eog_chs, eog_data, maps, band, samp_freq 
    
    # Plotting bad components
    print 'Marked %i component outlier(s) for subtraction (FASTER):'%len(bads)
    print bads
    if plot:
        ica.plot_sources(epochs_fs, exclude=[int(comp.replace('ICA', '')) for comp in bads])
    
    # Subtracting bad components (if chosen)
    if subtract:
        ica.apply(epochs, exclude=[int(comp.replace('ICA', '')) for comp in bads])
        ica.apply(epochs_fs, exclude=[int(comp.replace('ICA', '')) for comp in bads])
        epochs.apply_baseline(baseline)
        epochs_fs.apply_baseline(baseline)
    
    return epochs, epochs_fs, bads


# In[]
### Finding bad channels in single epochs (Stage IV.) ###

def find_bad_channels_in_epochs(epochs, epochs_fs, params=['var','median_grad','ampl','dev'], crit_z=3.0, interpolate=True, baseline=(-0.2, 0)):
    """
    Implements the fourth stage of FASTER: finds the channel outliers in single epochs.
    
    Parameters
    ----------
    epochs:     mne.io.Epochs
                The epoched data. Preferably returned by mnefaster.find_bad_components().
    params:     list
                The parameters to use. Default is to use them all.
                    'var':          The variance describing a given channel (Parameter 11.)
                    'median_grad':  The median gradient describing a given channel (Parameter 12.)
                    'ampl':         The maximum amplitude describing a given channel (Parameter 13.)
                    'dev':          The absolute deviation describing a given channel (Parameter 14.)
    crit_z:     float
                The critical z-score to use when determining outliers. Defaults to 3.0.
    interpolate:boolean
                Whether to interpolate channels marked as outliers in single epochs. Defaults to True.
    recursive:  boolean
                Whether to use recursive outliering method. Defaults to False.
    
    Returns
    -------
    epochs: mne.io.Epochs
            The epoched data (modified or unmodified).
    bads:   list of lists
            The labels of channels marked as outliers in single epochs.
    """
    # Defining global variables for multiprocessing
    global data, means
    
    # Getting requested parameters
    funcs = []
    if 'var' in params:
        funcs.append(_compute_channel_var)
    if 'median_grad' in params:
        funcs.append(_compute_channel_median_grad)
    if 'ampl' in params:
        funcs.append(_compute_channel_ampl)
    if 'dev' in params:
        funcs.append(_compute_channel_dev)

    # Getting the names of EEG channels
    eeg_chs = [epochs_fs.info['ch_names'][i] for i in mne.pick_types(epochs.info, meg=False, eeg=True)]
    
    # Getting epochs data
    epochs_data = epochs_fs.to_data_frame()
    
    # Getting channel means across epochs for Parameter 15.
    means = pd.DataFrame([np.mean(epochs_data[ch]) for ch in eeg_chs], index=eeg_chs).transpose() 
    
    # Iterating over epochs
    epochs_bads = []
    for i_epoch in set(epochs_data.index.get_level_values(level='epoch')):        
        # Getting data of a single epoch
        data = epochs_data.xs(i_epoch, level='epoch')
        # Defining channel outliers
        pool = Pool(workers)
        bads = []
        for func in funcs:
            chs = eeg_chs[:]
            values = list(pool.map(func, eeg_chs)) 
            # Outliering based on critical z-score
            z_chs = stats.zscore(values)
            bads_i = [i_ch for i_ch, z_ch in enumerate(z_chs) if abs(z_ch) > crit_z]
            bads += [chs[bad] for bad in bads_i if chs[bad] not in bads]
        epochs_bads.append(bads)  
        pool.close()
        pool.join()
    
    # Deleting global variables
    del data, means
        
    # Interpolating bad channels in single epochs (if chosen)
    def interpolate_bads_in_epochs(epochs):
        epochs_list = []
        for i in range(len(epochs)):
            ep = epochs[i]
            ep.info['bads'] = epochs_bads[i]
            if interpolate:
                ep.interpolate_bads(reset_bads=True)
            epochs_list.append(ep)
        epochs = mne.concatenate_epochs(epochs_list)
        epochs.apply_baseline(baseline)
        return epochs
    
    epochs = interpolate_bads_in_epochs(epochs)
    epochs_fs = interpolate_bads_in_epochs(epochs_fs)
    
    return epochs, epochs_fs, epochs_bads


# In[]
### Finding bad datasets (Stage V.) ###    
    
def find_bad_datasets(epochs_list, params=['ampl', 'var', 'dev'], crit_z=3.0, drop=True):
    """
    Implements the fifth step of FASTER: finds dataset outliers in the epoched data.
    
    Parameters
    ----------
    path:       string
                Path of the preprocessed individual datasets.
    ends:       string
                The end of filenames. Can be used to filter preprocessed datasets with different parameters.
    params:     list
                The parameters to use. Default is to use them all.
                    'ampl': The maximum channel amplitude describing an individual evoked response (Parameter 16.).
                    'var':  The mean of channel variances describing an individual evoked response (Parameter 17.)
                    'dev':  The mean of absolute channel deviations describing an individual evoked response (Parameter 18.)
    crit_z:     float
                The critical z-score to use when determining outliers. Defaults to 3.0.
    drop:       boolean
                Whether to modify 'epochs' by excluding the epochs marked as outliers. Defaults to True.
    recursive:  boolean
                Whether to use recursive outliering method. Defaults to False.
                
    Returns
    -------
    evokeds_list:   list
                    The evoked responses (mne.io.Evokeds) of the individual datasets.
    bads:           list
                    The indices of the datasets marked as outliers.
    """
    # Definig global variables for multiprocessing
    global data, means
    
    # Getting requested parameters
    funcs = []
    if 'ampl' in params:
        funcs.append(_compute_channel_ampl)
    if 'var' in params:
        funcs.append(_compute_channel_dev)
    if 'dev' in params:
        funcs.append(_compute_channel_var)
    
    # Creating individual ERPs from the preprocessed datasets 
    evokeds_list = [epochs.average() for epochs in epochs_list]
    
    # Getting the names of EEG channels
    eeg_chs = [evokeds_list[0].info['ch_names'][i] for i in mne.pick_types(evokeds_list[0].info, meg=False, eeg=True)]
    
    # Creating dataframes from individual ERPs
    evokeds_data_list = [evoked.to_data_frame() for evoked in evokeds_list]
    
    # Getting channel means across individual ERPs for Parameter 5.
    means = pd.DataFrame([np.mean([evoked[ch] for evoked in evokeds_data_list]) for ch in eeg_chs], index=eeg_chs).transpose()
    
    # Iterating over epochs
    evokeds_values = np.empty((0, len(funcs)))
    for i, data in enumerate(evokeds_data_list):
        # Computing all parameters for current epoch
        pool = Pool(workers)
        values = []
        for func in funcs:        
            values.append(np.mean(pool.map(func, eeg_chs)))
        evokeds_values = np.vstack([evokeds_values, np.asarray([values])]) 
        pool.close()
        pool.join()
        
    # Deleting global variables
    del data, means
    
    # Outliering based on critical z-score
    bads = []
    evokeds_z = stats.zscore(evokeds_values)
    bads += [i_ev for i_ev, z_ev in enumerate(evokeds_z) if any(abs(z) > crit_z for z in z_ev)]
            
    # Plotting bad datasets
    print 'Marked %i dataset outlier(s) for exclusion (FASTER):'%len(bads)
    print bads    
    
    # Dropping epoch outliers (if chosen)
    if drop:
        for bad in [evokeds_list[bad] for bad in bads]:
            evokeds_list.remove(bad)
    
    return evokeds_list, bads
