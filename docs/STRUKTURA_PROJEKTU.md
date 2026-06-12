# Struktura projektu

Ten plik opisuje, do czego sluza glowne foldery i pliki w projekcie EEG/BIAI.
Komendy najlepiej uruchamiac z katalogu glownego projektu: `C:\pliki\python\biai`.

## Katalog glowny

- `README.md` - krotki punkt startowy i linki do dokumentacji.
- `postepyprojektu.md` - dziennik zmian. Kazda kolejna zmiana w projekcie powinna byc tu dopisana.
- `.gitignore` - lista danych, wynikow i plikow tymczasowych, ktore nie powinny trafic do repo.

## `scripts/`

Kod Pythona do przetwarzania danych, budowania datasetow, trenowania modeli i agregacji wynikow.

### Budowanie datasetow

- `scripts/discover_visual_recall_sessions.py` - wyszukuje sesje visual recall w `dane/Wyniki` i zapisuje manifest sesji.
- `scripts/build_event_spectrogram_dataset.py` - buduje dataset spektrogramow wokol triggerow EEG.
- `scripts/build_event_epoch_dataset.py` - buduje dataset surowych epok EEG dla jednej sesji.
- `scripts/build_event_epoch_window_grid.py` - buduje wiele datasetow epok dla roznych okien czasowych.
- `scripts/build_multi_session_epoch_dataset.py` - buduje dataset surowych epok z wielu sesji/uczestnikow.
- `scripts/build_random_epoch_control_dataset.py` - buduje kontrolny dataset losowych okien EEG dopasowany do istniejacego datasetu eventowego.

Buildery epok zapisuja dodatkowo raporty QC:
- `epoch_qc.csv` - metryki jakosci kazdej kandydackiej epoki.
- `epoch_qc_summary.csv` - podsumowanie odrzucen, amplitud peak-to-peak i flatline.
- kolumny `qc_accepted`, `qc_reject_reason`, `qc_ptp_max_uv`, `qc_max_abs_uv` w `metadata.csv`.

### Trening i ewaluacja

- `scripts/train_eegnet.py` - trenuje EEGNet na surowych epokach; obsluguje split po seriach, uczestniku, obrazie i losowy oraz permutacje etykiet jako negative control.
- `scripts/run_eegnet_window_sweep.py` - uruchamia EEGNet dla wielu datasetow okien czasowych.
- `scripts/run_eegnet_participant_loo.py` - uruchamia leave-one-participant-out dla EEGNet.
- `scripts/train_epoch_bandpower_baseline.py` - klasyczny baseline logistyczny na pasmach mocy z surowych epok.
- `scripts/train_spectrogram_baseline.py` - prosty baseline klasyczny na spektrogramach.
- `scripts/train_spectrogram_cnn.py` - maly CNN na spektrogramach EEG.

### Agregacja wynikow

- `scripts/aggregate_eegnet_window_results.py` - zbiera wyniki sweepu okien czasowych.
- `scripts/aggregate_participant_loo_results.py` - zbiera wyniki leave-one-participant-out.

### Starsze / pomocnicze skrypty

- `scripts/process_eeg.py` - starszy pipeline pod VQ-VAE: preprocessing, ICA/smoothing i obrazy PSD.
- `scripts/AnalizaEEG1.py` - starszy skrypt analityczny z poczatku projektu.
- `scripts/Mapowanie nazw kanałów (Standard 10-20).py` - pomocniczy skrypt dotyczacy mapowania kanalow EEG.

## `docs/`

Dokumentacja, plany i materialy referencyjne.

- `docs/PLAN_SPEKTROGRAMY_I_SIECI.md` - aktualny plan eksperymentow ze spektrogramami, triggerami i EEGNet.
- `docs/WYNIKI_EKSPERYMENTOW.md` - zbiorcze wyniki najwazniejszych eksperymentow i kontroli.
- `docs/PIPELINE_DETALICZNY.md` - opis starszego, szczegolowego pipeline'u preprocessingowego.
- `docs/LOG_PROJEKTU.md` - starszy log projektu.
- `docs/plan.txt` - wczesniejszy plan prac.
- `docs/biai.pdf` - material zrodlowy / referencyjny.
- `docs/archive/README.mdgit` - stary pomocniczy plik README zachowany archiwalnie.

## `notebooks/`

Notebooki do eksploracji, prezentacji i dziennika prac.

- `notebooks/DETALICZNY_PIPELINE.ipynb` - notebook pokazujacy szczegolowy pipeline.
- `notebooks/DZIENNIK_PROJEKTU_EEG.ipynb` - notebook-dziennik prac EEG.
- `notebooks/WIZUALIZACJA_PIPELINE_DETALICZNA.ipynb` - notebook z wizualizacja pipeline'u.

## `manifests/`

Pliki opisujace zestawy sesji/danych.

- `manifests/visual_recall_sessions_manifest.csv` - manifest sesji visual recall: uczestnik, rep, EDF, CSV eventow, impedancja i liczba eventow `IMAGE_ON`.

## Dane i wygenerowane wyniki

Te katalogi zostaly zostawione w obecnych lokalizacjach, bo wiele skryptow i wynikow juz sie do nich odnosi.

- `dane/` - dane zrodlowe, w tym EDF, CSV z eventami i impedancja.
- `event_spectrogram_dataset*` - wygenerowane datasety spektrogramow.
- `event_epoch_dataset*` - wygenerowane datasety surowych epok dla jednej sesji.
- `event_epoch_window_grid*` - datasety dla roznych okien czasowych.
- `event_epoch_multisession_*` - dataset wielosesyjny.
- `baseline_results/` - wyniki klasycznych baseline'ow.
- `cnn_results/` - wyniki CNN na spektrogramach.
- `eegnet_results/` - wyniki pojedynczych uruchomien EEGNet.
- `eegnet_window_results*` - wyniki sweepow okien czasowych.
- `eegnet_multisession_results*` - wyniki leave-one-participant-out i wariantow multi-session.
- `eeg_dataset_vqvae/` - starszy dataset obrazow PSD pod VQ-VAE.
- `processing_results_v2/` - starsze wyniki preprocessingowe.

## Najczestsze komendy

Odkrycie sesji:

```powershell
python .\scripts\discover_visual_recall_sessions.py
```

Budowa datasetu multi-session:

```powershell
python .\scripts\build_multi_session_epoch_dataset.py
```

Budowa datasetu multi-session z odrzuceniem epok nieprzechodzacych QC:

```powershell
python .\scripts\build_multi_session_epoch_dataset.py --output-dir event_epoch_multisession_image_on_0_0p8_qc --drop-rejected
```

Budowa kontrolnego datasetu losowych okien:

```powershell
python .\scripts\build_random_epoch_control_dataset.py --reference-dataset-dir event_epoch_multisession_image_on_0_0p8 --output-dir event_epoch_random_control
```

Szybki trening EEGNet:

```powershell
python .\scripts\train_eegnet.py --dataset-dir event_epoch_multisession_image_on_0_0p8 --split participant --test-participant mole --epochs 12 --batch-size 256
```

Trening EEGNet tylko na epokach zaakceptowanych przez QC:

```powershell
python .\scripts\train_eegnet.py --dataset-dir event_epoch_multisession_image_on_0_0p8_qc --split participant --test-participant mole --epochs 12 --batch-size 256 --only-qc-accepted
```

Trening kontrolny z permutacja etykiet:

```powershell
python .\scripts\train_eegnet.py --dataset-dir event_epoch_multisession_image_on_0_0p8 --split participant --test-participant mole --label-control permute --epochs 12 --batch-size 256
```

Split jednoczesnie po uczestniku i niewidzianych obrazach:

```powershell
python .\scripts\train_eegnet.py --dataset-dir event_epoch_multisession_image_on_0_0p8 --split participant_image --test-participant mole --epochs 12 --batch-size 256
```

Baseline bandpower na surowych epokach:

```powershell
python .\scripts\train_epoch_bandpower_baseline.py --dataset-dir event_epoch_multisession_image_on_0_0p8 --split participant_image --test-participant mole
```

Leave-one-participant-out:

```powershell
python .\scripts\run_eegnet_participant_loo.py --dataset-dir event_epoch_multisession_image_on_0_0p8 --results-parent eegnet_multisession_results
```
