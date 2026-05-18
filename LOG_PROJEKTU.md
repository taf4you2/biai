# Log Projektu BIAI - Analiza EEG

## [2026-05-18] - Inicjalizacja i Preprocessing
- **Analiza dokumentacji**: Przeanalizowano `biai.pdf` pod kątem wymagań dla systemu DSI-24 i modelu VQ-VAE.
- **Konfiguracja środowiska**: Zainstalowano kluczowe biblioteki: `mne`, `pandas`, `matplotlib`, `numpy`, `scikit-learn`.
- **Analiza repozytorium**: Przegląd `AtomVQVAE` - model obrazowy (128x128), który zostanie zaadaptowany do sygnałów EEG.
- **Implementacja `process_eeg.py`**:
    - Wczytywanie danych EDF i CSV.
    - Automatyczne mapowanie nazw kanałów do standardu 10-20.
    - Odrzucanie kanałów o niskim SNR (X1, X2, X3).
    - Filtrowanie pasmowoprzepustowe (1-40 Hz).
    - Wykrywanie anomalii amplitudowych (próg 200 uV).
- **Wyniki**: Wykryto ponad 15 tys. punktów anomalnych w surowym sygnale.

## Planowane kroki:
1. Generowanie spektrogramów/obrazów 128x128 z sygnału EEG dla modelu AtomVQVAE.
2. Przygotowanie skryptu trenującego (Dataset Loader).
3. Detekcja anomalii na podstawie błędu rekonstrukcji VQ-VAE.
