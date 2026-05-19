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

## [2026-05-19] - Zaawansowany Pipeline i Przygotowanie Datasetu
- **Implementacja `DETALICZNY_PIPELINE.ipynb`**: Interaktywny notebook demonstrujący każdy krok przetwarzania.
- **Aktualizacja `process_eeg.py`**:
    - Dodano filtr Notch (50 Hz).
    - Wdrożono Common Average Reference (CAR).
    - Automatyczna korekcja artefaktów ocznych za pomocą ICA.
    - Wygładzanie czasowe filtrem Savitzky-Golay.
    - Generowanie spektrogramów 128x128 wygładzonych filtrem Gaussa.
- **Wyniki**: Wygenerowano 21 obrazów treningowych w `eeg_dataset_vqvae/` gotowych do użycia w modelu VQ-VAE.

## Planowane kroki:
1. Przygotowanie skryptu trenującego model VQ-VAE (AtomVQVAE) w PyTorch.
2. Detekcja anomalii na podstawie błędu rekonstrukcji.
3. Eksperymenty z parametrami wygładzania w celu optymalizacji stabilności modelu.
