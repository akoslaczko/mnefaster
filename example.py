#!/usr/bin/env python
# coding: utf-8
# In[ ]:
get_ipython().magic(u'matplotlib tk') # Needs to be run in an IPython console

# In[ ]:
import os
import gc
import string
import numpy as np
import pandas as pd
import mne
import mnefaster_dev2 as faster
from multiprocessing import freeze_support

# In[ ]:
def preprocess_individual_data(path, sessions=['dot', 'arabic'], event_type='targ', f_band=(0.1, 95), f_band_fs=(2, 45), ica_method='picard', tmin=-0.5, tmax=1.5, baseline=(-0.2, 0), crit_z=3.0, apply_notch=True, decimate=False, dc=10, drop_eog_events=False, eog_tresh=200, drop_errors=False, first_only=False):
    """Preprocessing on individual level"""
    
    ### Defining local functions
    def _set_montage(raw, mont):
        """Sets channel types for the GSN-Hydrocel-129 montage"""
        # Renaming channels
        raw.rename_channels(dict(zip(raw.ch_names, [ch_name.replace('EG ', '').replace('E0', 'E').replace('E0', 'E') for ch_name in raw.ch_names])))
#        raw.rename_channels({'E129':'Cz'})
        raw.rename_channels(dict(E129='Cz'))
        # Assembling channel lists
        stim_chs = [ch for ch in raw.ch_names if list(ch)[0] == 'n'] + ['STI 014'] + ['Cz']
        resp_chs = [ch for ch in raw.ch_names if list(ch)[0] == 'r']
        eog_chs = ['E8', 'E25', 'E125', 'E126', 'E127', 'E128'] # The official EOG channels
        misc_chs = []
        eeg_chs = [ch for ch in raw.info['ch_names'] if ch not in stim_chs + resp_chs + eog_chs + misc_chs]
        # Setting channel types in Raw
        ch_types = dict(zip(['stim', 'resp', 'eog', 'misc', 'eeg'], [stim_chs, resp_chs, eog_chs, misc_chs, eeg_chs]))
        for ch_type in ch_types:
            for ch in ch_types[ch_type]:
                raw.set_channel_types({ch:ch_type})
        # Setting montage
        raw.set_montage(mont)
        return raw

    
    def _read_evt(path, keep_cols=['B','E'], names={'B':'label', 'E':'session-onset'}):
        """Function to extract event information from Netstation .evt files"""
        # Reading evt file into a DataFrame object
        def _move_up_last_row(df):
            last = list(df.index)[len(df.index)-1]
            idx = [last] + df.index.drop(last).tolist()
            df = df.reindex(idx).reset_index().drop(['index'], axis=1)
            return df

        df = pd.read_csv(path, sep='\t', header=None, names=list(string.ascii_uppercase), skiprows=range(3), na_values=['',' '])              
        df = df.drop([x for x in df.columns if x not in keep_cols], axis=1).rename(columns=names) # to keep only specific columns
        df = _move_up_last_row(df)
        # Extracting variables
        parsed = pd.DataFrame(df['label'].str.split('_').tolist(), columns=['event_type', 'ref', 'targ', 'resp'])
        _, df['onset'] = np.asarray(df['session-onset'].str.split(' ',1).tolist()).T # to split 'session-onset' column
        df['onset'] = map(int, pd.to_timedelta(df['onset']) / np.timedelta64(1, 'ms')) # to convert 'onset' to int (ms)
        df = df.drop(['session-onset', 'label'], axis=1).reset_index(drop=True)
        df = pd.concat([df, parsed], axis=1)
        df = df.convert_objects(convert_numeric=True, copy=False).fillna(0)
        # Defining event categories
        df['notation'] = ['1' if max(df['ref']) == 9 else '2' for x in range(len(df))]
        sizes = sorted(set([int(evt[0])+int(evt[1]) if int(evt[1]) != 0 else 0 for evt in zip(df['ref'], df['targ'])]))
        distances = sorted(set([abs(int(evt[0])-int(evt[1])) if int(evt[1]) != 0 else 0 for evt in zip(df['ref'], df['targ'])]))
        df['size'] = ['2' if evt[0]+evt[1] > sizes[int(len(sizes)/2)] else '1' if evt[0]+evt[1] < sizes[int(len(sizes)/2)] else '0' for evt in zip(df['ref'], df['targ'])]
        df['distance'] = ['2' if abs(evt[0]-evt[1]) > distances[3] else '1' if abs(evt[0]-evt[1]) < distances[3]   else '0' for evt in zip(df['ref'], df['targ'])]
        df['event_id'] = [int(evt[0]+evt[1]+evt[2]) for evt in zip(df['notation'], df['size'], df['distance'])]
        return df
  
    
    def _process_session(session):
        """Function to process sessions data"""
        name = [dir_file for dir_file in os.listdir(path) if dir_file.startswith(sub) and dir_file.endswith('%s.raw' % session)][0] #TODO Raise error when len(name) != 1
        raw = mne.io.read_raw_egi(path + name, exclude=[])
        # Set montage & channel types
        mont = mne.channels.read_montage(kind='GSN-HydroCel-129')
        raw = _set_montage(raw, mont)
        # Reading event log
        log_data = _read_evt(path + sub + '_' + session + '.evt')
        log_data = log_data[log_data['event_type']==event_type]
        # Reading behavioral data
        behav = pd.read_csv(path + 'behav/' + sub + '.csv', header=None, names=['id', 'session', 'resp_code', 'ref', 'targ', 'resp_key', 'rt', 'error'])
        behav = behav[behav['id'] == sub]
        behav = behav[behav['session'] == session]
        ### Erroneous trials
        errors = [i for i, error in enumerate(behav['error']) if error == 1]
        ### Extreme RT trials
#        rts = map(float, behav['rt'])
#        badrts = [i for i, rt in enumerate(rts) if rt > np.mean(rts) + 2 * np.std(rts) or rt < np.mean(rts) - 2 * np.std(rts)]
#        errors = list(set(errors + badrts))
        # Defining events
        if event_type == 'targ':
            events = np.asarray([log_data['onset'], [0]*len(log_data), log_data['event_id']]).T
            if session == 'arabic':
                sid = '1'
            elif session == 'dot':
                sid = '2'
            event_id = {session+'/ss/sd':int(sid+'11'), 
                        session+'/ls/sd':int(sid+'21'),
                        session+'/ms/sd':int(sid+'01'),
                        session+'/ss/md':int(sid+'10'), 
                        session+'/ls/md':int(sid+'20'),
                        session+'/ss/ld':int(sid+'12'), 
                        session+'/ls/ld':int(sid+'22'),
                        session+'/ms/ld':int(sid+'02')}
        elif event_type == 'ref':
            events = np.asarray([log_data['onset'], [0]*len(log_data), log_data['ref']]).T
            if session == 'arabic':
                event_id = {session+'/one':1,
                            session+'/two':2,
                            session+'/three':3,
                            session+'/four':4,
                            session+'/five':5,
                            session+'/six':6,
                            session+'/seven':7,
                            session+'/eight':8,
                            session+'/nine':9}
            elif session == 'dot':
                event_id = {session+'/one':5,
                            session+'/two':10,
                            session+'/three':15,
                            session+'/four':20,
                            session+'/five':25,
                            session+'/six':30,
                            session+'/seven':35,
                            session+'/eight':40,
                            session+'/nine':45}

        # FASTER stage I.
        raw, raw_fs, bad_channels = faster.find_bad_channels(raw, mont, f_band=f_band, f_band_fs=f_band_fs, apply_notch=apply_notch, crit_z=crit_z)
        # FASTER stage II.
        epochs, epochs_fs, bad_epochs = faster.find_bad_epochs(raw, raw_fs, events, event_id, tmin=tmin, tmax=tmax, baseline=baseline, crit_z=crit_z, decimate=decimate, dc=dc, drop_eog_events=drop_eog_events, eog_tresh=eog_tresh, drop_errors=drop_errors, errors=errors)
        # Fitting ICA
        epochs, epochs_fs, ica = faster.fit_ica(epochs, epochs_fs, n_interpolated=len(bad_channels), ica_method=ica_method, baseline=baseline)
        # FASTER stage III.
        epochs, epochs_fs, bad_components = faster.find_bad_components(epochs, epochs_fs, ica, baseline=baseline, crit_z=crit_z)
        # FASTER stage IV.
        epochs, epochs_fs, epochs_bads = faster.find_bad_channels_in_epochs(epochs, epochs_fs, baseline=baseline, crit_z=crit_z)
        return epochs
    
    
    def _process_participant(sub):
        """Function to process participants data"""
        epochs_list = []
        for session in sessions:
            epochs_list.append(_process_session(session))
        # Concatenating blocks
        epochs = mne.concatenate_epochs(epochs_list)
        epochs.apply_baseline(baseline)
        # Plotting ERP
        evoked = epochs.average()
        #evoked.plot_joint(title=sub, times=[0.120, 0.180, 0.230, 0.350, 0.600], ts_args=dict(gfp=True, zorder='unsorted', ylim=dict(eeg=[-10, 10]), time_unit='ms'), topomap_args=dict(vmin=-10, vmax=10, time_unit='ms'))
        evoked.plot_joint(title=sub)
        # Saving processed Epochs object to disc 
        epochs.save(path + 'test/' + sub + '-[preprocessed]' + '[%s]' %(event_type) + '-[z_%.1f]' %(crit_z) + '-epo.fif')
    
    
    ### Preprocessing the data of individual subjects
    participants = sorted(list(set([dir_file.split('_')[0] for dir_file in os.listdir(path) if not dir_file.endswith('.evt') and not dir_file in ['preprocessed', 'behav']])))
    print participants
    
    for sub in participants:
        if first_only:
            if sub == participants[1]:
                break
        _process_participant(sub)
        gc.collect() # To clean swap as well

# In[ ]:
path = '/media/akkus/Data/EEG/DistSize/'

# In[ ]:
if __name__ == '__main__':
    freeze_support() # Freeze support for Windows (multiprocessing) 
    
    preprocess_individual_data(path, crit_z=3, event_type='targ', f_band=(0.1, 90), f_band_fs=(4, 90), ica_method='extended-infomax', apply_notch=True, baseline=(-0.2, 0), tmin=-0.5, tmax=1.5, decimate=False, dc=2, drop_eog_events=False, eog_tresh=120, drop_errors=True, first_only=True)
