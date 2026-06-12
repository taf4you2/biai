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

## [2026-06-11] - Przejscie na dataset zdarzeniowy i EEGNet
- **Zmiana glownego kierunku**: Projekt przeszedl ze starszego toru VQ-VAE/PSD na klasyfikacje reakcji EEG wokol triggerow zadania visual recall.
- **Dataset zdarzeniowy**:
    - Dodano budowanie epok EEG wokol `IMAGE_ON`.
    - Wybrano praktyczne okno `0.0 .. 0.8 s` jako najlepszy kompromis po sweepie okien.
    - Zbudowano dataset wielosesyjny `event_epoch_multisession_image_on_0_0p8`.
- **Skala danych**:
    - `abc`: 1320 epok.
    - `Bear`: 2200 epok.
    - `fghx`: 2640 epok.
    - `mole`: 1980 epok.
    - `Reshi`: 1980 epok.
    - Razem: 10120 epok, 11 klas.
- **Modelowanie**:
    - Glownym modelem stal sie EEGNet na surowych epokach `1 x 21 x 480`.
    - Historyczny leave-one-participant-out dawal ok. `18.83%` sredniej trafnosci, przy poziomie losowym `9.09%`.
- **Metodologia**:
    - Dodano wybor checkpointu po walidacji zamiast po tescie.
    - Dodano raporty QC mapowania eventow.
    - Dodano split po `image_id`.
    - Dodano negative controls: permutacja etykiet i losowe okna EEG.

## [2026-06-11] - Porzadkowanie repozytorium
- **Nowa struktura**:
    - `scripts/` - skrypty Pythona.
    - `docs/` - dokumentacja i pliki referencyjne.
    - `notebooks/` - notebooki.
    - `manifests/` - manifesty sesji.
- **Dokumentacja**:
    - Dodano `docs/STRUKTURA_PROJEKTU.md` z opisem folderow i roli plikow.
    - `postepyprojektu.md` zostal ustawiony jako glowny dziennik zmian.
    - `README.md` zostal uproszczony jako punkt startowy.

## [2026-06-12] - Pelne kontrole wiarygodnosci wynikow
- **Bazowy eksperyment LOO z walidacja**:
    - Dataset: `event_epoch_multisession_image_on_0_0p8`.
    - Split: leave-one-participant-out.
    - Checkpoint wybierany po walidacji.
    - Wynik: mean test accuracy `17.19%`.
- **LOO z normalizacja per uczestnik i samplerem**:
    - Uruchomiono `--normalization participant --balanced-sampler category_participant`.
    - Wynik: mean test accuracy `17.76%`.
    - Interpretacja: lekka poprawa wzgledem bazowego LOO, ok. `+0.58 pp`.
- **Permutacja etykiet**:
    - Uruchomiono LOO z `--label-control permute`.
    - Wynik: mean test accuracy `8.93%`.
    - Interpretacja: kontrola spada do poziomu losowego.
- **Losowe okna EEG**:
    - Zbudowano dataset `event_epoch_random_control` z 10120 losowych epok.
    - Uruchomiono LOO na losowych oknach.
    - Wynik: mean test accuracy `9.20%`.
    - Interpretacja: przypadkowe fragmenty sygnalu nie niosa uzytecznej informacji o klasie.
- **Split po niewidzianych obrazach**:
    - Uruchomiono trening z `--split image`.
    - Wynik: test accuracy `20.80%`.
    - Interpretacja: model generalizuje ponad losowo takze na niewidziane `image_id`.
- **Najwazniejszy wniosek**:
    - Kontrole negatywne sa blisko poziomu losowego `9.09%`, a normalne eksperymenty zostaja powyzej losowego poziomu.
    - Aktualne wyniki nie wygladaja na prosty przeciek etykiet, dryft sesji albo przypadkowy kontekst czasowy.
- **Szczegoly**:
    - Pelna tabela znajduje sie w `docs/WYNIKI_EKSPERYMENTOW.md`.
    - Chronologiczny dziennik zmian znajduje sie w `postepyprojektu.md`.

## Aktualne planowane kroki
1. Powtorzyc LOO z `--normalization participant --balanced-sampler category_participant` w trybie z walidacja.
2. Dodac split mieszany: held-out participant + held-out `image_id`.
3. Dodac raport QC epok: amplitudy, peak-to-peak, odrzucanie artefaktow i liczba odrzuconych probek per uczestnik.
4. Porownac aktualne wyniki z wariantami normalizacji i samplerow w `docs/WYNIKI_EKSPERYMENTOW.md`.

## Stare planowane kroki (historyczne)
1. Przygotowanie skryptu trenującego model VQ-VAE (AtomVQVAE) w PyTorch.
2. Detekcja anomalii na podstawie błędu rekonstrukcji.
3. Eksperymenty z parametrami wygładzania w celu optymalizacji stabilności modelu.
