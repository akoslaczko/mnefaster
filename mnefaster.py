# -*- coding: utf-8 -*-
"""
FASTER
"""


### Importing module dependencies ###

import mne
import numpy as np
import pandas as pd
from scipy import stats
from scipy import signal
from multiprocessing import Pool




### Hurst exponent function from 'eegfaster' ###

def _hurst(x):
    """FASTER [Nolan2010]_ implementation of the Hurst Exponent.
    Parameters
    ----------
    x : numpy.ndarray
        Vector with the data sequence.
    Returns
    -------
    h : float
        Computed hurst exponent
    #-SPHINX-IGNORE-#
    References
    ----------
    [Nolan2010] H. Nolan, R. Whelan, and R.B. Reilly. Faster: Fully automated
    statistical thresholding for eeg artifact rejection. Journal of
    Neuroscience Methods, 192(1):152-162, 2010.
    #-SPHINX-IGNORE-#
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
    



### Function to get correct polar angle distances from the reference electrode ###

def _get_correct_thetas(mont, ref_ch, eeg_chs):
    """Get correct angular distances from the reference electrode"""
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
    
    
 

### Functions for single parameters (multiprocessing.Pool.map() needs top level functions which are pickleable). ###
    
def _get_channels_corr(ch_pair, data=None):
    """Get Pearson's correlation coefficient between two channels"""
    # Picking up variables from the child process
    if data is None:
        data = data_child
    # Getting the correlation coefficient (absolute)
    chs_corr = abs(stats.pearsonr(data[ch_pair[0]], data[ch_pair[1]])[0])
    return (ch_pair, chs_corr)

def _get_channel_mcorr(ch, corr_dict=None):
    """Get the mean correlation coefficient describing a given channel"""
    # Picking up variables from the child process
    if corr_dict is None:
        corr_dict = corr_dict_child
    # Getting the mean correlation coefficient
    ch_mcorr = np.mean([corr_dict[ch_pair] for ch_pair in corr_dict.keys() if ch in ch_pair])
    return ch_mcorr

def _get_channel_var(ch, data=None):
    """Get the variance describing a given channel"""
    # Picking up variables from the child process
    if data is None:
        data = data_child
    # Getting the variance
    ch_var = np.var(data[ch])
    return ch_var
                         
def _get_channel_hurst(ch, data=None):
    """Get the Hurst exponent describing a given channel"""
    # Picking up variables from the child process
    if data is None:
        data = data_child
    # Getting the Hurst exponent
    ch_hurst = _hurst(np.asarray(data[ch]))
    return ch_hurst
    
def _get_channel_ampl(ch, data=None):
    """Get the amplitude describing a given channel"""
    # Picking up variables from the child process
    if data is None: 
        data = data_child
    # Getting the amplitude
    ch_ampl = max(data[ch]) - min(data[ch])
    return ch_ampl

def _get_channel_dev(ch, data=None, means=None):
    """Get the deviance describing a given channel"""
    # Picking up variables from the child process
    if data is None: 
        data = data_child
    if means is None: 
        means = means_child
    # Getting the deviance (absolute)
    ch_dev = abs(means[ch] - np.mean(data[ch]))
    return ch_dev   

def _get_channel_median_grad(ch, data=None):
    """Get the median gradient describing a given channel"""
    # Picking up variables from the child process
    if data is None:
        data = data_child
    # Getting the median gradient
    ch_med_grad = np.median(np.diff(data[ch])) 
    return ch_med_grad

def _get_component_eog_corr(comp, data=None, misc=None, misc_data=None):
    """Get the maximum correlation coefficient with EOG channels describing a given ICA component"""
    # Picking up variables from the child process
    if data is None:
        data = data_child
    if misc is None:
        misc = misc_child
    if misc_data is None:
        misc_data = misc_data_child
    # Getting the maximum EOG correlation coefficient
    c_max_eog = max([abs(stats.pearsonr(data[comp], misc_data[ch])[0]) for ch in misc])
    return c_max_eog

def _get_component_spatial_kurt(comp, maps=None):
    """Get the spatial kurtosis describing a given ICA component"""
    # Picking up variables from the child process
    if maps is None:
        maps = maps_child
    # Getting the spatial kurtosis
    c_kurt = stats.kurtosis(maps[comp])
    return c_kurt

def _get_component_mean_slope(comp, data=None, band=None):
    """Get the mean slope in the frequency band describing a given ICA component"""
    # Picking up variables from the child process
    if data is None:
        data = data_child
    if band is None:
        band = band_child
    # Getting the mean slope in the frequency band
    freqs, spectra = signal.welch(data[comp], window='hamming', fs=1000) # TODO inherit fs from a higher level object
    c_mean_slope = np.mean(np.diff(10*np.log10([spectra[i] for i, f in enumerate(freqs) if f > band[0] and f < band[1]])))
    return c_mean_slope

def _get_component_hurst(comp, data=None):
    """Get the Hurst exponent describing a given ICA component"""
    # Picking up variables from the child process
    if data is None:
        data = data_child
    # Getting Hurst exponent
    c_hurst = _hurst(np.asarray(data[comp]))
    return c_hurst

def _get_component_median_grad(comp, data=None):
    """Get the median gradient describing a given ICA component"""
    # Picking up variables from the child process
    if data is None:
        data = data_child
    # Getting the median gradient
    c_med_grad = np.median(np.diff(data[comp])) 
    return c_med_grad





  
##### The actual stages of FASTER #####    
    
### Finding bad channels in raw data (Stage I.) ###

def find_bad_channels(raw, mont, params=['mean_corr','var','hurst'], crit_z=3, f_band=(4,30), ref_ch='E11', reset_ref=True, apply_filter=True, interpolate=True):
    
    # Getting requested parameters
    funcs = []
    if 'mean_corr' in params:
        funcs.append(_get_channel_mcorr)
    if 'var' in params:
        funcs.append(_get_channel_var)
    if 'hurst' in params:
        funcs.append(_get_channel_hurst)
    
    # Getting the names of EEG channels
    eeg_chs = [raw.info['ch_names'][i] for i in mne.pick_types(raw.info, meg=False, eeg=True)]
    
    # Getting data from the mne.io.Raw object
    data_parent = raw.to_data_frame()
    
    # Setting the Fz electrode as reference (if chosen)    
    if reset_ref:
        raw.load_data()
        raw, _ = mne.io.set_eeg_reference(raw, [ref_ch])
        raw.set_channel_types({ref_ch:'misc'})
    
    # Applying band-pass filter (if chosen)
    if apply_filter:
        raw.load_data() # TODO: check whether this is necessary 
        raw.filter(f_band[0], f_band[1])
        raw.plot_psd(area_mode='range', fmax=f_band[1]+10 , picks=mne.pick_types(raw.info, meg=False, eeg=True))
    
    # Getting channel correlations for Parameter 1.
    if 'mean_corr' in params:
        # Getting all possible channel pairings
        ch_pairs = []
        for ch_x in eeg_chs:
            ch_pairs += [(ch_x, ch_y) for ch_y in eeg_chs if ch_x != ch_y and (ch_y, ch_x) not in ch_pairs]
            
        # Getting correlation coefficients
        def initializer(): # Passing necessary variables to the child processes
            global data_child
            data_child = data_parent
        pool = Pool(None, initializer, ())
        corr_dict_parent = dict(pool.map(_get_channels_corr, ch_pairs))
        # Closing Pool
        pool.close()
        pool.join()
    
    # Getting correct polar angles from the reference electrode
    thetas = _get_correct_thetas(mont, ref_ch, eeg_chs)
    
    # Finding channel outliers in raw data
    def initializer(): # Passing necessary variables to the child processes
        global data_child, corr_dict_child
        data_child = data_parent
        corr_dict_child = corr_dict_parent
    pool = Pool(None, initializer, ())
    
    # Defining channel outliers based on critical z-score
    bads = []
    for func in funcs:
        if func == _get_channel_mcorr or func == _get_channel_var:
            values = pool.map(func, eeg_chs)
            # Quadratic correction for distance from reference electrode (in case of mean channel correlation and variance)
            p = np.polyfit(thetas, values, 2)
            fitcurve = np.polyval(p, thetas)
            z_chs = stats.zscore(values - fitcurve)
        else:
            z_chs = stats.zscore(pool.map(func, eeg_chs))
        bads += [eeg_chs[i_ch] for i_ch, z_ch in enumerate(z_chs) if abs(z_ch) > crit_z and eeg_chs[i_ch] not in bads]
    
    # Closing Pool
    pool.close()
    pool.join()
    
    # Plot bad channels
    print 'Marked %i channel outlier(s) for interpolation (FASTER):' %len(bads)
    print bads
    raw.info['bads'] = bads
    raw.plot(bad_color='red')
    
    # Interpolating bad channels (if chosen)
    if interpolate:
        raw.interpolate_bads(reset_bads=True)
    
    return raw, bads
    
    

    
    
### Finding bad epochs (Stage II.) ###

def find_bad_epochs(raw, events, event_id, params=['ampl','dev','var'], crit_z=3, tmin=-0.5, tmax=1.5, baseline=(-0.2,0), drop=True):
    
    # Getting requested parameters
    funcs = []
    if 'ampl' in params:
        funcs.append(_get_channel_ampl)
    if 'dev' in params:
        funcs.append(_get_channel_dev)
    if 'var' in params:
        funcs.append(_get_channel_var)
    
    # Getting the names of EEG channels
    eeg_chs = [raw.info['ch_names'][i] for i in mne.pick_types(raw.info, meg=False, eeg=True)]
    
    # Creating Epochs object from Raw
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=baseline, picks=mne.pick_types(raw.info, meg=False, eeg=True, misc=True)).drop_bad()

    # Getting epochs data
    epochs_data = epochs.to_data_frame()
    
    # Getting channel means across epochs for Parameter 5.
    means_parent = pd.DataFrame([np.mean(epochs_data[ch]) for ch in eeg_chs], index=eeg_chs).transpose()
    
    # Iterating over epochs
    epochs_values = np.empty((0,3))
    for i_epoch in set(epochs_data.index.get_level_values(level='epoch')):
        
        # Getting data of single epoch
        data_parent = epochs_data.xs(i_epoch, level='epoch')
        
        # Computing all parameters for current epoch
        def initializer(): # Passing necessary variables to the child processes
            global data_child, means_child
            data_child = data_parent
            means_child = means_parent
        pool = Pool(None, initializer, ())
        
        values = []
        for func in funcs:        
            values.append(np.mean(pool.map(func, eeg_chs)))
        epochs_values = np.vstack([epochs_values, np.asarray([values])])
        
        # Closing Pool   
        pool.close()
        pool.join()
    
    # Defining outlier epochs based on critical z-score
    z_epochs = stats.zscore(epochs_values)        
    bads = [i_epoch for i_epoch, z_epoch in enumerate(z_epochs) if any(abs(z) > crit_z for z in z_epoch)]
    
    # Plotting bad epochs
    print 'Marked %i epoch outlier(s) for exclusion (FASTER):'%len(bads)
    print bads    
    
    # Dropping epoch outliers (if chosen)
    if drop:
        epochs.drop(bads)
    
    return epochs, bads
    



### Fitting ICA with predefined parameters ###

def fit_ica(epochs, n_interpolated, ref_ch='E11', method='infomax', reset_ref=True):
    
    # Getting the names of EEG channels
    eeg_chs = [epochs.info['ch_names'][i] for i in mne.pick_types(epochs.info, meg=False, eeg=True)]
    # Getting the names of misc channels
    misc_chs = [epochs.info['ch_names'][i] for i in mne.pick_types(epochs.info, meg=False, misc=True)]
    
    # Setting average as reference if epochs is still Fz-referenced (if chosen)
    if reset_ref == True:
        epochs.load_data()
        epochs, _ = mne.io.set_eeg_reference(epochs, eeg_chs)
        
    # Temporarily setting misc channels as EEG to include them in the ICA   
    for ch in misc_chs: 
        epochs.set_channel_types({ch:'eeg'}) 
        
    # Defining the correct number of PCA components   
    c_pca = int(min(np.floor(np.sqrt(len(epochs.to_data_frame().index)/25)), epochs.info['nchan'] - n_interpolated))
    
    # Running ICA with the chosen method
    ica = mne.preprocessing.ICA(method=method, n_components=c_pca-1, max_pca_components=c_pca)
    ica.fit(epochs)
     
    # Resetting the misc channels
    for ch in misc_chs: 
        epochs.set_channel_types({ch:'misc'})
        
    return epochs, ica






### Finding bad components (Stage III.) ###

def find_bad_components(epochs, ica, params=['eog_corr','spatial_kurt','mean_slope','hurst','median_grad'], crit_z=3, subtract=True):
    
    # Getting requested parameters
    funcs = []
    if 'eog_corr' in params:
        funcs.append(_get_component_eog_corr)
    if 'spatial_kurt' in params:
        funcs.append(_get_component_spatial_kurt)
    if 'mean_slope' in params:
        funcs.append(_get_component_mean_slope)
    if 'hurst' in params:
        funcs.append(_get_component_hurst)
    if 'median_grad' in params:
        funcs.append(_get_component_median_grad)

    # Getting the names of misc channels
    misc_parent = [epochs.info['ch_names'][i] for i in mne.pick_types(epochs.info, meg=False, misc=True)]
    
    # Getting temporal data of ICA components
    data_parent = ica.get_sources(epochs).to_data_frame()
    
    # Getting the names of ICA components
    comps = data_parent.columns.values
    
    # Getting spatial data of ICA components (jona-sassenhagen)
    maps_parent = pd.DataFrame(np.dot(ica.mixing_matrix_[:, list(range(ica.n_components_))].T, ica.pca_components_[:ica.n_components_]), index=comps).transpose()

    # Getting epochs data (for EOG correlations)
    misc_data_parent = epochs.to_data_frame()
    
    # Finding component outliers
    def initializer(): # Passing necessary variables to the child processes
        global data_child, maps_child, misc_child, misc_data_child, band_child
        data_child = data_parent
        maps_child = maps_parent
        misc_child = misc_parent
        misc_data_child = misc_data_parent
        band_child = (epochs.info['highpass'], epochs.info['lowpass'])   
    pool = Pool(None, initializer, ())
    
    # Defining component outliers based on critical z-score
    bads = []
    for func in funcs:
        z_comps = stats.zscore(pool.map(func, comps))
        bads += [comps[i_comp] for i_comp, z_comp in enumerate(z_comps) if abs(z_comp) > crit_z and comps[i_comp] not in bads]
        
    # Closing Pool
    pool.close()
    pool.join()
    
    # Plotting bad components
    print 'Marked %i component outlier(s) for subtraction (FASTER):'%len(bads)
    print bads
    ica.plot_sources(epochs, exclude=[int(comp.replace('ICA ', '')) - 1 for comp in bads])
    
    # Subtracting bad components (if chosen)
    if subtract:
        ica.apply(epochs, exclude=[int(comp.replace('ICA ', '')) - 1 for comp in bads])
    
    return epochs, bads





### Finding bad channels in single epochs (Stage IV.) ###

def find_bad_channels_in_epochs(epochs, params=['var','median_grad','ampl','dev'], crit_z=3, interpolate=True):
    
    # Getting requested parameters
    funcs = []
    if 'var' in params:
        funcs.append(_get_channel_var)
    if 'median_grad' in params:
        funcs.append(_get_channel_median_grad)
    if 'ampl' in params:
        funcs.append(_get_channel_ampl)
    if 'dev' in params:
        funcs.append(_get_channel_dev)

    # Getting the names of EEG channels
    eeg_chs = [epochs.info['ch_names'][i] for i in mne.pick_types(epochs.info, meg=False, eeg=True)]
    
    # Getting epochs data
    epochs_data = epochs.to_data_frame()
    
    # Getting channel means across epochs for Parameter 15.
    means_parent = pd.DataFrame([np.mean(epochs_data[ch]) for ch in eeg_chs], index=eeg_chs).transpose() 
    
    # Iterating over epochs
    epochs_bads = []
    for i_epoch in set(epochs_data.index.get_level_values(level='epoch')):
        
        # Getting data of a single epoch
        data_parent = epochs_data.xs(i_epoch, level='epoch')
        
        # Finding channel outliers in single epochs
        def initializer():# Passing necessary variables to the child processes
            global data_child, means_child
            data_child = data_parent
            means_child = means_parent
        pool = Pool(None, initializer, ())
        
        # Defining channel outliers based on critical z-score
        bads = []
        for func in funcs:
            z_chs = stats.zscore(pool.map(func, eeg_chs))
            bads += [eeg_chs[i_ch] for i_ch, z_ch in enumerate(z_chs) if abs(z_ch) > crit_z and eeg_chs[i_ch] not in bads]
        epochs_bads.append(bads)
        
        # Closing Pool   
        pool.close()
        pool.join()
        
    # Interpolating bad channels in single epochs (if chosen)
    evoked = []
    for i, ev in enumerate(epochs.iter_evoked()):
        ev.info['bads'] = epochs_bads[i]
        if interpolate:
            ev.interpolate_bads(reset_bads=True)
        evoked.append(ev)
    evoked = mne.combine_evoked(evoked, weights='equal')
        
    return evoked, epochs_bads
