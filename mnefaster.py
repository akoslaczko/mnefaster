# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 13:51:08 2016
"""
import mne
import pandas as pd
import numpy as np
from scipy import stats
from scipy import signal
from multiprocessing import Pool


### Hurst exponent function from 'eegfaster' ###

# This is actually an exact translation of the author's matlab code, which supposed to be much faster than other "full-blown" implementation of the Hurst exponent.
def hurst(x):
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





### FASTER ###

# Default arguments
crit_z = 3
tmin = -0.5  # start of each epoch (500ms before the trigger)
tmax = 1.5  # end of each epoch (1500ms after the trigger)
baseline = (-0.2, 0)  # means from the first instant to t = - 200ms (as suggested by Nolan et al. (2010))


# Functions for single parameters (multiprocessing.Pool.map() needs top level functions which are pickleable)
def get_channels_corr(ch_pair, data=None):
    if not data: data = df
    chs_corr = stats.pearsonr(data[ch_pair[0]], data[ch_pair[1]])[0]
    return (ch_pair, chs_corr)

def get_channel_mcorr(ch, c_dict=None): # TODO: Quadratic correction for distance from reference electrode!
    if not c_dict: c_dict = corr_dict
    ch_mcorr = np.mean([c_dict[ch_pair] for ch_pair in c_dict.keys() if ch in ch_pair])
    return ch_mcorr

def get_channel_var(ch, data=None): # TODO: Quadratic correction for distance from reference electrode!
    if not data: data = df
    ch_var = np.var(data[ch])
    return ch_var
                         
def get_channel_hurst(ch, data=None):
    if not data: data = df
    ch_hurst = hurst(np.asarray(data[ch]))
    return ch_hurst

def get_channel_ampl(ch, data=None):
    if not data: data = df
    ch_ampl = max(data[ch]) - min(data[ch])
    return ch_ampl

def get_channel_dev(ch, data=None, mean=None):
    if not data: data = df
    if not mean: mean = ch_means[ch]
    ch_dev = mean - np.mean(data[ch])
    return ch_dev

def get_channel_median_grad(comp, data=None): #NEW
    if not data: data = df
    ch_med_grad = np.median(np.diff(data[comp])) 
    # median(diff(EEG.icaact(u,:)))
    return ch_med_grad

def get_component_eog_corr(comp, data=None, misc=None, misc_data=None):
    if not data: data = df_ica
    if not misc: misc = misc_chs
    if not misc_data: misc_data = df_epochs
    c_max_eog = max([abs(stats.pearsonr(data[comp], misc_data[ch])[0]) for ch in misc])
    return c_max_eog

def get_component_spatial_kurt(comp, data=None):
    if not data: data = df_map
    c_kurt = stats.kurtosis(data[comp])
    # kurt(EEG.icawinv(:,u))
    return c_kurt

def get_component_mean_slope(comp, data=None, band=None):
    if not data: data = df_ica
    if not band: band = f_band
    freqs, spectra = signal.welch(data[comp], window='hamming', fs=1000)
    c_mean_slope = np.mean(np.diff(10*np.log10([spectra[i] for i, f in enumerate(freqs) if f > band[0] and f < band[1]])))
    # pwelch(x,window,noverlap,f,fs) returns the two-sided Welch PSD estimates at the frequencies specified in the vector, f. The vector, f, must contain at least 2 elements. The frequencies in f are in cycles per unit time. The sampling frequency, fs, is the number of samples per unit time. If the unit of time is seconds, then f is in cycles/sec (Hz).
    # [spectra(u,:) freqs] = pwelch(EEG.icaact(u,:),[],[],(EEG.srate),EEG.srate)
    # mean(diff(10*log10(spectra(u,find(freqs>=lpf_band(1),1):find(freqs<=lpf_band(2),1,'last')))))
    return c_mean_slope

def get_component_hurst(comp, data=None):
    if not data: data = df_ica
    c_hurst = hurst(np.asarray(data[comp]))
    return c_hurst

def get_component_median_grad(comp, data=None):
    if not data: data = df_ica
    c_med_grad = np.median(np.diff(data[comp])) 
    # median(diff(EEG.icaact(u,:)))
    return c_med_grad




# Defining parameter-lists for stages
params_bad_channels = [get_channel_mcorr, get_channel_var, get_channel_hurst] # Parameters 1, 2, & 3 [Stage I.] (Nolan et al., 2010)

params_bad_epochs = [get_channel_ampl, get_channel_dev, get_channel_var] # Parameters 4, 5, & 6 [Stage II.] (Nolan et al., 2010)

params_bad_components = [get_component_eog_corr, get_component_spatial_kurt, get_component_mean_slope, get_component_hurst, get_component_median_grad] # Parameters 7, 8, 9, 10 & 11 [Stage III.] (Nolan et al., 2010)

params_bad_channels_in_epochs = [get_channel_var, get_channel_median_grad, get_channel_ampl, get_channel_dev] # Parameters 12, 13, 14, & 15 [Stage IV.] (Nolan et al., 2010)




# Finding bad channels (Stage I.)
def find_bad_channels(raw, montage, crit_z=crit_z, f_band=[4,30], ref_ch='E11', reset_ref=True, apply_filter=True, interpolate=True):
    eeg_chs = [raw.info['ch_names'][i] for i in mne.pick_types(raw.info, meg=False, eeg=True)]
    
    def set_custom_ref(raw):
        raw.load_data()
        raw, _ = mne.io.set_eeg_reference(raw, [ref_ch])
        raw.set_channel_types({ref_ch:'misc'})
        return raw
        
    if reset_ref == True:
        raw = set_custom_ref(raw)
    
    def filter_data(raw):
        raw.load_data()
        raw.filter(f_band[0], f_band[1])
        raw.plot_psd(area_mode='range', fmax=f_band[1]+10 , picks=mne.pick_types(raw.info, meg=False, eeg=True))
        return raw
    
    if apply_filter == True:
        raw = filter_data(raw)
    
    def get_corr_dict(raw):
        ch_pairs = [] # Getting all possible channel pairings
        for ch_x in eeg_chs:
            ch_pairs += [(ch_x, ch_y) for ch_y in eeg_chs if ch_x != ch_y and (ch_y, ch_x) not in ch_pairs]
        df_parent = raw.to_data_frame()
        def initializer(): # Initializing Pool for multiprocessing
            global df # Setting up global variables for the child processes
            df = df_parent
        pool = Pool(None, initializer, ())
        corr_dict = dict(pool.map(get_channels_corr, ch_pairs)) # Getting all possible channel-correlations
        # Closing Pool
        pool.close()
        pool.join()
        return corr_dict, df_parent
    
    corr_dict_parent, df_parent = get_corr_dict(raw) # Getting channel-correlation dictionary for Parameter 1 (passing 'df_parent' too)
    
    def get_polar_angles(montage, ref_ch=ref_ch, eeg_chs=eeg_chs):
        # Getting channel locations from montage
        df_mont = pd.DataFrame(mont.pos, index=mont.ch_names, columns=['X', 'Y', 'Z'])
        # Getting original polar angles
        thetas = pd.DataFrame([np.arctan2(np.sqrt(ch[1]['X']**2 + ch[1]['Y']**2), ch[1]['Z']) for ch in df_mont.iterrows()], index=mont.ch_names, columns=['theta'])
        # Rotating the sensor-space to match the vertex and the origo->ref_channel vector
        theta_rot = (360 * np.pi / 180) - thetas.loc[ref_ch]['theta']
        mat_rot = np.array([[1,0,0], [0, np.cos(theta_rot), -np.sin(theta_rot)], [0, np.sin(theta_rot), np.cos(theta_rot)]])
    
        rotated = np.empty((0,3))
        for ch in df_mont.iterrows():
            rotated = np.vstack((rotated, np.array(np.dot(np.asarray(ch[1]), mat_rot))))
        rotated = pd.DataFrame(rotated, index=mont.ch_names, columns=['X', 'Y', 'Z'])
    
        # Getting corrected polar angles
        thetas = [np.arctan2(np.sqrt(ch[1]['X']**2 + ch[1]['Y']**2), ch[1]['Z']) for ch in rotated.iterrows() if ch[0] in eeg_chs]
        return thetas
    
    thetas = get_polar_angles(montage)
    
    def initializer(): # Initializing Pool for multiprocessing
        global df, corr_dict # Setting up global variables for the child processes
        df = df_parent
        corr_dict = corr_dict_parent
    pool = Pool(None, initializer, ())
    # Computing parameters
    bads = []
    for param in params_bad_channels:
        if param == get_channel_mcorr or param == get_channel_var:
            values = pool.map(param, eeg_chs)
            # Quadratic correction for distance from reference electrode
            p = np.polyfit(thetas, values, 2)
            fitcurve = np.polyval(p, thetas)
            z_channels = stats.zscore(values - fitcurve)
        else:
            z_channels = stats.zscore(pool.map(param, eeg_chs))
        # Defining channel outliers
        bads += [eeg_chs[i_ch] for i_ch, z_ch in enumerate(z_channels) if abs(z_ch) > crit_z and eeg_chs[i_ch] not in bads]
    # Closing Pool
    pool.close()
    pool.join()
    
    # Plot bad channels
    print 'Interploating %i bad channels (FASTER):' %len(bads)
    print bads
    raw.info['bads'] = bads
    raw.plot(bad_color='red')
    
    if interpolate == True:
        raw.interpolate_bads(reset_bads=True)
    
    return raw, bads





# Finding bad epochs (Stage II.)
def find_bad_epochs(raw, events, event_id, crit_z=crit_z, tmin=tmin, tmax=tmax, baseline=baseline, drop=True):
    eeg_chs = [raw.info['ch_names'][i] for i in mne.pick_types(raw.info, meg=False, eeg=True)]
    
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=baseline, picks=mne.pick_types(raw.info, meg=False, eeg=True, misc=True)).drop_bad() # Creating Epochs object from Raw

    df_epochs = epochs.to_data_frame() # Get epochs data
    ch_means_parent = pd.DataFrame([np.mean(df_epochs[ch]) for ch in eeg_chs], index=eeg_chs).transpose() # Get channel means across epochs for Parameter 5
    
    # Iterating over epochs
    values_epochs = np.empty((0,3))
    for i_epoch in set(df_epochs.index.get_level_values(level='epoch')):
        df_parent = df_epochs.xs(i_epoch, level='epoch')
        def initializer(): # Initializing Pool for multiprocessing
            global df, ch_means # Setting up global variables for the child processes
            df = df_parent
            ch_means = ch_means_parent
        pool = Pool(None, initializer, ())
        # Computing parameters for each epoch
        values = []
        for param in params_bad_epochs:        
            values.append(np.mean(pool.map(param, eeg_chs)))
        values_epochs = np.vstack([values_epochs, np.asarray([values])])
        # Closing Pool   
        pool.close()
        pool.join()
    
    # Defining epoch outliers   
    z_epochs = stats.zscore(values_epochs)        
    bads = [i_epoch for i_epoch, z_epoch in enumerate(z_epochs) if any(abs(z) > crit_z for z in z_epoch)]
    
    # Dropping epoch outliers (optional)
    if drop == True:
        print 'Dropping %i epochs (FASTER):'%len(bads)
        print bads
        epochs.drop(bads)
    
    return epochs, bads





# Fitting ICA with predefined parameters (Nolan et al., 2010)
def fit_ica(epochs, n_interpolated, ref_ch='E11', method='infomax', reset_ref=True):
    
    eeg_chs = [epochs.info['ch_names'][i] for i in mne.pick_types(epochs.info, meg=False, eeg=True)]
    misc_chs = [epochs.info['ch_names'][i] for i in mne.pick_types(epochs.info, meg=False, misc=True)]
    
    if reset_ref == True: # Setting average as reference if epochs is still Fz-referenced (default)
        epochs.load_data()
        #epochs, _ = mne.io.set_eeg_reference(epochs, []) # Deactivating Fz reference projection (not sure whether this is is necessary)
        epochs, _ = mne.io.set_eeg_reference(epochs, eeg_chs) # Applying average reference projection (the built in method uses the eog channels as well?! Using eeg channels only for computing average reference...)
        
        
    for ch in misc_chs: # Temporarily setting misc channels as EEG to include them in the ICA
        epochs.set_channel_types({ch:'eeg'}) 
        
    c_pca = int(min(np.floor(np.sqrt(len(epochs.to_data_frame().index)/25)), epochs.info['nchan'] - n_interpolated)) # Defining the correct number of PCA components (see Nolan et al. (2010))
    
    # Running ICA with the chosen method
    ica = mne.preprocessing.ICA(method=method, n_components=c_pca-1, max_pca_components=c_pca)
    ica.fit(epochs)
        
    for ch in misc_chs: # Re-setting misc channels
        epochs.set_channel_types({ch:'misc'})
        
    return ica, epochs



# Finding bad components (Stage III.)
def find_bad_components(ica, epochs, crit_z=crit_z, subtract=True):
    
    df_ica_parent = ica.get_sources(epochs).to_data_frame()
    ica_comps = df_ica_parent.columns.values
    
    def get_ica_map(ica, components=None): # Function suggested by jona-sassenhagen
        """Get ICA topomap for components"""
        if components is None:
            components = list(range(ica.n_components_))
        maps = np.dot(ica.mixing_matrix_[:, components].T, ica.pca_components_[:ica.n_components_])
        # EEG.icawinv = pinv(EEG.icaweights*EEG.icasphere);
        return maps
    
    df_map_parent = pd.DataFrame(get_ica_map(ica), index=ica_comps).transpose() #TODO check if this is the correct way
    df_epochs_parent = epochs.to_data_frame()
    misc_chs_parent = [epochs.info['ch_names'][i] for i in mne.pick_types(epochs.info, meg=False, misc=True)]
    
    def initializer(): # Initializing Pool for multiprocessing
        global df_ica, df_map, df_epochs, f_band, misc_chs # Setting up global variables for the child processes
        df_ica = df_ica_parent
        df_map = df_map_parent
        df_epochs = df_epochs_parent
        misc_chs = misc_chs_parent
        f_band = (epochs.info['highpass'], epochs.info['lowpass'])
        
    pool = Pool(None, initializer, ())
    # Computing parameters
    bads = []
    for param in params_bad_components:
        z_comps = stats.zscore(pool.map(param, ica_comps))
        # Defining component outliers
        bads += [ica_comps[i_comp] for i_comp, z_comp in enumerate(z_comps) if abs(z_comp) > crit_z and ica_comps[i_comp] not in bads]
        
    # Closing Pool
    pool.close()
    pool.join()
    
    # Plot components
    bads_index = [int(comp.replace('ICA ', '')) - 1 for comp in bads]
    ica.plot_sources(epochs, exclude=bads_index)
    
    # Subtracting bad components
    if subtract == True:
        print 'Subtracting %i bad components (FASTER):'%len(bads)
        print bads
        ica.apply(epochs, exclude=bads_index)
    
    return epochs, bads




# Finding bad channels in epochs (Stage IV.)
def find_bad_channels_in_epochs(epochs, crit_z=crit_z, interpolate=True):
    
    eeg_chs = [epochs.info['ch_names'][i] for i in mne.pick_types(epochs.info, meg=False, eeg=True)]
    df_epochs = epochs.to_data_frame() # Get epochs data
    ch_means_parent = pd.DataFrame([np.mean(df_epochs[ch]) for ch in eeg_chs], index=eeg_chs).transpose() # Get channel means across epochs for Parameter 5
    
    # Iterating over epochs
    bads_in_epochs = []
    for i_epoch in set(df_epochs.index.get_level_values(level='epoch')):
        df_parent = df_epochs.xs(i_epoch, level='epoch')
        def initializer(): # Initializing Pool for multiprocessing
            global df, ch_means # Setting up global variables for the child processes
            df = df_parent
            ch_means = ch_means_parent
        pool = Pool(None, initializer, ())
        # Computing parameters for each epoch
        bads = []
        for param in params_bad_channels_in_epochs:
            z_channels = stats.zscore(pool.map(param, eeg_chs))
            # Defining channel outliers in each epoch
            bads += [eeg_chs[i_ch] for i_ch, z_ch in enumerate(z_channels) if abs(z_ch) > crit_z and eeg_chs[i_ch] not in bads]
        bads_in_epochs.append(bads)
        # Closing Pool   
        pool.close()
        pool.join()
        
    # Interpolating bad channels in single epochs (if chosen)
    evoked = []
    for i, ev in enumerate(epochs.iter_evoked()):
        ev.info['bads'] = bads_in_epochs[i]
        if interpolate == True:
            ev.interpolate_bads()
        evoked.append(ev)
    evoked = mne.combine_evoked(evoked, weights='equal')
        
    return evoked, bads_in_epochs