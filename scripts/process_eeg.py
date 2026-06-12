import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter
import os

# ==========================================================
# CONFIGURATION
# ==========================================================
CSV_PATH = 'dane/NBackExample_imp.csv'
EDF_PATH = 'dane/Alpha1_raw.edf'
OUTPUT_DIR = 'processing_results_v2'
DATASET_DIR = 'eeg_dataset_vqvae'

# Pipeline Parameters
REJECT_THRESHOLD = 0.50
L_FREQ = 1.0
H_FREQ = 40.0
NOTCH_FREQ = 50.0
EPOCH_DURATION = 2.0  # seconds
SAVGOL_WINDOW = 15
SAVGOL_POLY = 2
GAUSS_SIGMA = 0.7
IMG_SIZE = (128, 128)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

def run_advanced_pipeline():
    print(f"--- Step 1: Loading and Mapping ---")
    # Load CSV bad channels
    df_imp = pd.read_csv(CSV_PATH, skiprows=6)
    bad_channels_csv = []
    for col in df_imp.columns:
        if col == 'Time': continue
        low_snr_ratio = (df_imp[col].astype(str).str.strip() == 'Low_SNR').sum() / len(df_imp)
        if low_snr_ratio > REJECT_THRESHOLD:
            bad_channels_csv.append(col)
    
    # Load EDF
    raw = mne.io.read_raw_edf(EDF_PATH, preload=True, verbose=False)
    mapping = {ch: ch.split('-')[0].split(':')[0].strip() for ch in raw.ch_names}
    raw.rename_channels(mapping)
    
    # Set Montage
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='ignore')
    
    # Match bads
    bads_for_mne = []
    for bad_ch in bad_channels_csv:
        matched = [ch for ch in raw.ch_names if bad_ch == ch]
        bads_for_mne.extend(matched)
    raw.info['bads'] = bads_for_mne
    
    # Pick EEG and Filter
    raw.pick_types(eeg=True, exclude=[]) # Keep bads for interpolation/CAR context if needed
    raw.filter(l_freq=L_FREQ, h_freq=H_FREQ, verbose=False)
    raw.notch_filter(NOTCH_FREQ, verbose=False)
    
    print(f"--- Step 2: Referencing (CAR) ---")
    raw.set_eeg_reference('average', projection=False, verbose=False)
    
    print(f"--- Step 3: Artifact Correction (ICA) ---")
    # We use a copy for ICA to avoid issues with bad channels
    ica = mne.preprocessing.ICA(n_components=15, random_state=97, method='fastica', verbose=False)
    ica.fit(raw, picks='eeg', reject_by_annotation=True)
    
    # Find EOG artifacts based on frontal channels
    print(f"Available channels for ICA: {raw.ch_names}")
    
    # Try to find Fp1/Fp2 or similar
    fp_channels = [ch for ch in raw.ch_names if 'Fp1' in ch or 'Fp2' in ch]
    print(f"Detected frontal channels for EOG correction: {fp_channels}")
    
    if fp_channels:
        eog_indices, _ = ica.find_bads_eog(raw, ch_name=fp_channels, verbose=False)
        ica.exclude = eog_indices
    else:
        print("WARNING: No Fp1/Fp2 channels found. Skipping automatic EOG correction.")
    
    print(f"Excluded ICA components: {ica.exclude}")
    
    raw_clean = raw.copy()
    ica.apply(raw_clean, verbose=False)

    print(f"--- Step 4: Smoothing (Savitzky-Golay) ---")
    data = raw_clean.get_data()
    data_savgol = savgol_filter(data, window_length=SAVGOL_WINDOW, polyorder=SAVGOL_POLY, axis=1)
    raw_savgol = raw_clean.copy()
    raw_savgol._data = data_savgol

    print(f"--- Step 5: Epoching and Spectral Analysis ---")
    epochs = mne.make_fixed_length_epochs(raw_savgol, duration=EPOCH_DURATION, preload=True, verbose=False)
    # Reject bad epochs
    epochs.drop_bad(reject=dict(eeg=150e-6), verbose=False)
    print(f"Generated {len(epochs)} clean epochs.")

    # Process each epoch for VQ-VAE
    print(f"--- Step 6: Generating Dataset Images (128x128) ---")
    for i, epoch in enumerate(epochs):
        # Compute PSD using Welch
        spectrum = epochs[i].compute_psd(method='welch', fmin=1, fmax=40, n_fft=256, verbose=False)
        psd_data = spectrum.get_data()[0] # (channels, freqs)
        
        # Log-transform and Normalize
        img_raw = np.log10(psd_data + 1e-10)
        img_norm = (img_raw - img_raw.min()) / (img_raw.max() - img_raw.min() + 1e-10)
        
        # Apply Gaussian Smoothing to the image
        img_final = gaussian_filter(img_norm, sigma=GAUSS_SIGMA)
        
        # Resize/Interpolate to exactly 128x128
        # We use matplotlib to save as a clean image without axes
        plt.figure(figsize=(1.28, 1.28), dpi=100) # 1.28 * 100 = 128 pixels
        plt.imshow(img_final, aspect='auto', cmap='magma', interpolation='bilinear')
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        
        img_path = os.path.join(DATASET_DIR, f"epoch_{i:04d}.png")
        plt.savefig(img_path, pad_inches=0)
        plt.close()

    print(f"\nProcessing Complete!")
    print(f"Visual results: '{OUTPUT_DIR}'")
    print(f"VQ-VAE Dataset: '{DATASET_DIR}' ({len(epochs)} images)")

    # Save a final verification plot
    fig = epochs.average().plot(show=False)
    fig.savefig(os.path.join(OUTPUT_DIR, 'final_evoked_check.png'))
    plt.close()

if __name__ == "__main__":
    run_advanced_pipeline()
