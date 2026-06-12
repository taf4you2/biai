import mne
import numpy as np
import matplotlib.pyplot as plt

file_path = "./data/Alpha1_raw.edf"
raw = mne.io.read_raw_edf(file_path, preload=True, infer_types=True)
mapping = {ch: ch.split('-')[0].split(':')[0].strip() for ch in raw.ch_names}
raw.rename_channels(mapping)

ch_types = {}
for ch in raw.ch_names:
    if ch in ['X1', 'X2', 'X3', 'CM', 'Ax', 'Ay', 'Az']:
        ch_types[ch] = 'misc'  
    elif ch in ['Trigger', 'Event']:
        ch_types[ch] = 'stim'
    else:
        ch_types[ch] = 'eeg'
raw.set_channel_types(ch_types)

montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage, on_missing='ignore')

print(raw.info)

raw.plot_sensors(kind='topomap', show_names=True);
 