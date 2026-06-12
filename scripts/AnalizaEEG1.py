import mne
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# 1. AUTOMATYCZNE WYKRYWANIE ZŁYCH KANAŁÓW Z PLIKU CSV
# ==========================================================

# Ścieżka do pliku CSV zawierającego informacje
# o jakości sygnału dla każdego kanału EEG.
csv_path = 'NBackExample_imp.csv'

# Wczytanie pliku CSV z pominięciem pierwszych 6 wierszy.
# Pominięte wiersze zawierają metadane, takie jak szczegóły
# nagrania, informacje o urządzeniu lub ustawienia eksportu.
# Siódmy wiersz zawiera właściwe nagłówki kolumn.
df_imp = pd.read_csv(csv_path, skiprows=6)

# Lista przechowująca kanały sklasyfikowane jako niepoprawne.
bad_channels = []

# Maksymalny dopuszczalny udział próbek oznaczonych jako Low_SNR.
# Kanały przekraczające tę wartość zostaną oznaczone jako złe.
reject_threshold = 0.50

# Iteracja po wszystkich kolumnach w pliku CSV.
for col in df_imp.columns:

    # Pominięcie kolumny czasu, ponieważ nie zawiera danych EEG.
    if col == 'Time':
        continue

    # Obliczenie udziału próbek oznaczonych jako "Low_SNR".
    #
    # Kroki przetwarzania:
    # 1. Konwersja wartości do typu tekstowego.
    # 2. Usunięcie zbędnych spacji na początku i końcu.
    # 3. Porównanie wartości z etykietą "Low_SNR".
    # 4. Zliczenie pasujących wpisów.
    # 5. Podzielenie przez całkowitą liczbę wierszy.
    low_snr_ratio = (
        df_imp[col]
        .astype(str)
        .str.strip()
        == 'Low_SNR'
    ).sum() / len(df_imp)

    # Oznaczenie kanału jako złego, jeśli udział Low_SNR
    # przekracza ustalony próg odrzucenia.
    if low_snr_ratio > reject_threshold:
        bad_channels.append(col)

# Wyświetlenie automatycznie odrzuconych kanałów.
print(f"Based on the CSV file, automatically rejected channels: {bad_channels}")

# ==========================================================
# 2. WCZYTYWANIE DANYCH EEG Z PLIKU EDF
# ==========================================================

# Ścieżka do pliku EDF zawierającego surowe dane EEG.
edf_path = 'Alpha1_raw.edf'

# Wczytanie pliku EDF do obiektu MNE Raw.
#
# preload=True powoduje natychmiastowe załadowanie danych
# do pamięci, co przyspiesza późniejsze filtrowanie i analizę.
raw = mne.io.read_raw_edf(edf_path, preload=True)

# ----------------------------------------------------------
# DOPASOWANIE NAZW KANAŁÓW MIĘDZY PLIKAMI CSV I EDF
# ----------------------------------------------------------
#
# Nazwy kanałów w pliku CSV i EDF mogą się różnić.
# Przykład:
#   CSV: "X1"
#   EDF: "EEG X1"
#
# Poniższa sekcja wyszukuje nazwy kanałów EDF
# zawierające etykietę kanału z pliku CSV.
# ----------------------------------------------------------

# Lista przechowująca nazwy kanałów dokładnie
# w formacie używanym w pliku EDF.
bads_for_mne = []

# Iteracja po wszystkich automatycznie odrzuconych kanałach.
for bad_ch in bad_channels:

    # Wyszukiwanie częściowych dopasowań nazw kanałów EDF.
    matched = [ch for ch in raw.ch_names if bad_ch in ch]

    # Dodanie wszystkich pasujących nazw do listy.
    bads_for_mne.extend(matched)

# Zapisanie informacji o złych kanałach w obiekcie MNE Raw.
# Funkcje MNE automatycznie pomijają te kanały
# podczas wielu operacji analitycznych.
raw.info['bads'] = bads_for_mne

# Wyświetlenie końcowej listy złych kanałów używanej przez MNE.
print(f"Updated MNE Raw object. Bad channels: {raw.info['bads']}")

# ==========================================================
# 3. CZYSZCZENIE I ANALIZA SYGNAŁU
# ==========================================================

# Zastosowanie filtru pasmowoprzepustowego do sygnału EEG.
#
# Dolna granica: 1 Hz
#   Usuwa powolny dryft sygnału i artefakty ruchowe.
#
# Górna granica: 40 Hz
#   Usuwa szumy wysokoczęstotliwościowe i artefakty mięśniowe.
#
# Kanały oznaczone jako złe pozostają wykluczone z przetwarzania.
raw.filter(l_freq=1.0, h_freq=40.0)

# Obliczenie gęstości widmowej mocy (PSD).
#
# PSD przedstawia moc sygnału w funkcji częstotliwości
# i pomaga identyfikować dominujące rytmy EEG oraz zakłócenia.
#
# fmax=40 ogranicza zakres wyświetlanych częstotliwości do 40 Hz.
fig_psd = raw.compute_psd(fmax=40.0).plot()

# Dodanie opisowego tytułu do wykresu PSD.
fig_psd.suptitle('Power Spectral Density – Cleaned Signal')

# Wyświetlenie wszystkich wygenerowanych wykresów matplotlib.
# Jest to wymagane podczas uruchamiania skryptu poza
# środowiskami notebooków interaktywnych.
plt.show()

import os

# ==========================================================
# 4. GENEROWANIE SPEKTROGRAMÓW
# ==========================================================

# Utworzenie listy zawierającej wyłącznie poprawne kanały EEG.
# Kanały oznaczone jako złe są pomijane.
clean_channels = [
    ch for ch in raw.ch_names
    if ch not in raw.info['bads']
]

# Pobranie częstotliwości próbkowania z metadanych nagrania.
# Wartość ta jest wymagana do obliczeń czasowo-częstotliwościowych.
sfreq = raw.info['sfreq']

# Folder wyjściowy dla wygenerowanych obrazów spektrogramów.
folder_results = 'spektrogramy_wyniki'

# Utworzenie folderu wyjściowego, jeśli jeszcze nie istnieje.
# exist_ok=True zapobiega błędom, gdy folder już istnieje.
os.makedirs(folder_results, exist_ok=True)

# Wyświetlenie informacji o postępie działania programu.
print(f"Generowanie spektrogramów dla {len(clean_channels)} kanałów. Proszę czekać...")

# Generowanie spektrogramu dla każdego poprawnego kanału EEG.
for ch_name in clean_channels:

    # Pobranie próbek sygnału oraz odpowiadających im znaczników czasu
    # dla wybranego kanału.
    data, times = raw.get_data(
        picks=ch_name,
        return_times=True
    )

    # raw.get_data zwraca tablicę 2D nawet dla pojedynczego kanału.
    # Pobranie pierwszego wiersza w celu uzyskania tablicy 1D.
    channel_data = data[0]

    # Utworzenie nowego okna wykresu dla bieżącego spektrogramu.
    plt.figure(figsize=(10, 5))

    # Generowanie spektrogramu.
    #
    # Fs:
    #   Częstotliwość próbkowania.
    #
    # NFFT:
    #   Liczba próbek używana w każdym oknie FFT.
    #   Większe wartości poprawiają rozdzielczość częstotliwościową.
    #
    # noverlap:
    #   Liczba nakładających się próbek między oknami.
    #   Większe nakładanie poprawia ciągłość czasową.
    #
    # cmap:
    #   Mapa kolorów używana do wizualizacji.
    Pxx, freqs, bins, im = plt.specgram(
        channel_data,
        Fs=sfreq,
        NFFT=512,
        noverlap=256,
        cmap='viridis'
    )

    # Dodanie tytułu wykresu zawierającego nazwę kanału.
    plt.title(f'EEG Spectrogram - Channel: {ch_name}')

    # Opis osi wykresu.
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')

    # Ograniczenie wyświetlanego zakresu częstotliwości
    # do zakresu istotnego dla EEG.
    plt.ylim(0, 40)

    # Dodanie skali kolorów reprezentującej intensywność sygnału.
    plt.colorbar(label='Power / Intensity (dB)')

    # Automatyczne dopasowanie odstępów między elementami wykresu.
    plt.tight_layout()

    # Utworzenie bezpiecznej nazwy pliku poprzez zastąpienie
    # spacji znakami podkreślenia.
    file_name = f"{folder_results}/spectrogram_{ch_name.replace(' ', '_')}.png"

    # Zapisanie spektrogramu jako obrazu PNG.
    plt.savefig(file_name)

    # Zamknięcie bieżącego wykresu w celu zwolnienia pamięci.
    # Jest to szczególnie ważne podczas przetwarzania wielu kanałów.
    plt.close()

# Komunikat końcowy informujący o zakończeniu działania.
print("Done")