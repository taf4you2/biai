import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================================
# CONFIGURATION
# ==========================================================
CSV_PATH = 'dane/NBackExample_imp.csv'
EDF_PATH = 'dane/Alpha1_raw.edf'
REJECT_THRESHOLD = 0.50
L_FREQ = 1.0
H_FREQ = 40.0
OUTPUT_DIR = 'processing_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_clean_eeg():
    print(f"--- Step 1: Loading CSV and detecting bad channels ---")
    # Load CSV, skipping metadata lines
    df_imp = pd.read_csv(CSV_PATH, skiprows=6)
    
    bad_channels_csv = []
    for col in df_imp.columns:
        if col == 'Time': continue
        
        low_snr_ratio = (
            df_imp[col].astype(str).str.strip() == 'Low_SNR'
        ).sum() / len(df_imp)
        
        if low_snr_ratio > REJECT_THRESHOLD:
            bad_channels_csv.append(col)
    
    print(f"Automatically rejected channels from CSV: {bad_channels_csv}")

    print(f"\n--- Step 2: Loading EDF and mapping channels ---")
    # Load EDF
    raw = mne.io.read_raw_edf(EDF_PATH, preload=True, infer_types=True)
    
    # Mapping names (from 'Mapowanie nazw kanałów.py')
    mapping = {ch: ch.split('-')[0].split(':')[0].strip() for ch in raw.ch_names}
    raw.rename_channels(mapping)
    
    # Set channel types
    ch_types = {}
    for ch in raw.ch_names:
        if ch in ['X1', 'X2', 'X3', 'CM', 'Ax', 'Ay', 'Az']:
            ch_types[ch] = 'misc'
        elif ch in ['Trigger', 'Event']:
            ch_types[ch] = 'stim'
        else:
            ch_types[ch] = 'eeg'
    raw.set_channel_types(ch_types)
    
    # Set standard 10-20 montage
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='ignore')

    # Match CSV bad channels with MNE channel names
    bads_for_mne = []
    for bad_ch in bad_channels_csv:
        matched = [ch for ch in raw.ch_names if bad_ch == ch]
        bads_for_mne.extend(matched)
    
    raw.info['bads'] = bads_for_mne
    print(f"Final bad channels list: {raw.info['bads']}")

    print(f"\n--- Step 3: Filtering and Anomaly Detection ---")
    # Filtering
    raw.filter(l_freq=L_FREQ, h_freq=H_FREQ)
    
    # Simple Anomaly Detection: Amplitude thresholding
    # We look for segments where amplitude exceeds a certain threshold (e.g., 200uV)
    # in clean channels.
    clean_picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    data, times = raw.get_data(picks=clean_picks, return_times=True)
    
    # Threshold for anomaly (in Volts, 200uV = 200e-6)
    threshold = 200e-6
    anomalies = np.where(np.abs(data) > threshold)
    
    num_anomalies = len(anomalies[0])
    print(f"Detected {num_anomalies} anomalous points (exceeding {threshold*1e6} uV)")
    
    if num_anomalies > 0:
        unique_anom_channels = [raw.ch_names[clean_picks[i]] for i in np.unique(anomalies[0])]
        print(f"Anomalies detected in channels: {unique_anom_channels}")

    # Plot PSD
    fig_psd = raw.compute_psd(fmax=H_FREQ).plot()
    fig_psd.savefig(os.path.join(OUTPUT_DIR, 'psd_cleaned.png'))
    plt.close()
    
    print(f"\n--- Processing complete. Results saved in '{OUTPUT_DIR}' ---")
    return raw

if __name__ == "__main__":
    raw_processed = load_and_clean_eeg()
    print(raw_processed.info)
