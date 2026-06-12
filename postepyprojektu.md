# Postepy projektu

Ten plik jest glownym dziennikiem zmian i postepow w projekcie.

Zasada pracy:
- kazda kolejna zmiana w kodzie, danych, wynikach eksperymentow albo dokumentacji ma byc dopisana do tego pliku;
- wpis powinien zawierac date, krotki opis zmiany, dotkniete pliki oraz sposob weryfikacji, jesli byla wykonana;
- jezeli zmiana nie byla testowana, trzeba to jawnie zaznaczyc.

## 2026-06-11

- Utworzono ten dziennik jako stale miejsce zapisywania postepow projektu.
- Ustalono, ze przyszle zmiany maja byc dokumentowane tutaj.

### Zmiany pipeline EEG

- Dodano walidacje w `train_eegnet.py`: model wybiera najlepszy checkpoint po wyniku walidacyjnym, a test jest liczony dopiero raz na koncu.
- Dodano argumenty `--val-size` i `--val-split` do treningu EEGNet oraz skryptow uruchamiajacych sweep/LOO.
- Dodano split `--split image` w `train_eegnet.py`, zeby testowac generalizacje na niewidzianych `image_id`.
- Poprawiono rozwiazywanie sciezek do epok w `train_eegnet.py`, tak aby dzialaly tez sciezki z podfolderami uczestnikow.
- Dodano raport `event_alignment_qc.csv` do builderow datasetow: spektrogramow, pojedynczych epok, siatki okien i datasetu multi-session.
- Do `session_summary.csv` datasetu multi-session dodano pola QC dla docelowego eventu.
- Zaktualizowano agregatory wynikow EEGNet, zeby obslugiwaly nowe pola walidacji i byly kompatybilne ze starymi summary.

Dotkniete pliki:
- `train_eegnet.py`
- `run_eegnet_participant_loo.py`
- `run_eegnet_window_sweep.py`
- `aggregate_participant_loo_results.py`
- `aggregate_eegnet_window_results.py`
- `build_event_spectrogram_dataset.py`
- `build_event_epoch_dataset.py`
- `build_event_epoch_window_grid.py`
- `build_multi_session_epoch_dataset.py`
- `postepyprojektu.md`

Weryfikacja:
- `python -m py_compile train_eegnet.py run_eegnet_participant_loo.py run_eegnet_window_sweep.py aggregate_participant_loo_results.py aggregate_eegnet_window_results.py build_event_spectrogram_dataset.py build_event_epoch_dataset.py build_multi_session_epoch_dataset.py build_event_epoch_window_grid.py`
- Smoke trening: `python train_eegnet.py --dataset-dir event_epoch_window_grid_fine\image_on_0_0p8 --output-dir eegnet_validation_smoke --split series --epochs 1 --batch-size 512 --cpu`
- Smoke trening przeszedl: train/validation/test = `1100/440/440`, best epoch = `1`, final test accuracy = `0.1364`; katalog smoke usunieto po sprawdzeniu.
- Agregatory uruchomiono na istniejacych wynikach: `eegnet_multisession_results_participant_balanced` i `eegnet_window_results_fine`.

### Porzadkowanie struktury plikow

- Przeniesiono skrypty Pythona do folderu `scripts/`.
- Przeniesiono dokumentacje i materialy referencyjne do folderu `docs/`.
- Przeniesiono notebooki do folderu `notebooks/`.
- Przeniesiono manifest sesji do folderu `manifests/`.
- Zostawiono w katalogu glownym `README.md`, `.gitignore` i `postepyprojektu.md` jako pliki startowe projektu.
- Dodano `docs/STRUKTURA_PROJEKTU.md`, czyli mape folderow i plikow z opisem ich roli.
- Zaktualizowano `README.md`, zeby wskazywal najwazniejsze miejsca w projekcie.
- Zaktualizowano domyslne sciezki manifestu w `scripts/discover_visual_recall_sessions.py` i `scripts/build_multi_session_epoch_dataset.py`.
- Zaktualizowano `scripts/run_eegnet_window_sweep.py` i `scripts/run_eegnet_participant_loo.py`, zeby po przeniesieniu wywolywaly skrypty przez sciezke folderu `scripts/`.
- Poprawiono przyklady komend w `docs/PLAN_SPEKTROGRAMY_I_SIECI.md`.

Dotkniete lokalizacje:
- `scripts/`
- `docs/`
- `notebooks/`
- `manifests/`
- `README.md`
- `postepyprojektu.md`

Weryfikacja:
- Skompilowano wszystkie pliki `*.py` z folderu `scripts/`.
- Sprawdzono `--help` dla `scripts/train_eegnet.py`, `scripts/run_eegnet_window_sweep.py` i `scripts/run_eegnet_participant_loo.py`.
- Smoke manifest: `python .\scripts\discover_visual_recall_sessions.py --output .\manifests\visual_recall_sessions_manifest_smoke.csv`; plik smoke usunieto po sprawdzeniu.
- Agregacja wynikow: `python .\scripts\aggregate_eegnet_window_results.py --results-dir eegnet_window_results_fine --output eegnet_window_results_fine\window_comparison_summary.csv`.
- Smoke trening: `python .\scripts\train_eegnet.py --dataset-dir event_epoch_window_grid_fine\image_on_0_0p8 --output-dir eegnet_reorg_smoke --split series --epochs 1 --batch-size 512 --cpu`.
- Smoke trening przeszedl: train/validation/test = `1100/440/440`, best epoch = `1`, final test accuracy = `0.1364`; katalog smoke usunieto po sprawdzeniu.

### Negative controls dla EEGNet

- Dodano w `scripts/train_eegnet.py` argument `--label-control permute`, ktory deterministycznie miesza `image_category` przy zadanym seedzie.
- Dodano propagacje `--label-control` przez `scripts/run_eegnet_window_sweep.py` i `scripts/run_eegnet_participant_loo.py`.
- Dodano `label_control` do agregatorow wynikow, zeby wyniki kontrolne nie mieszaly sie z normalnymi.
- Dodano `scripts/build_random_epoch_control_dataset.py`, ktory tworzy kontrolny dataset losowych okien EEG dopasowany liczba probek, etykietami i dlugoscia epoki do istniejacego datasetu eventowego.
- Zaktualizowano `docs/STRUKTURA_PROJEKTU.md` o nowe skrypty i komendy negative-control.
- Dodano `event_epoch_random_control*/` i `eegnet_label_permute*/` do `.gitignore`.

Dotkniete pliki:
- `scripts/train_eegnet.py`
- `scripts/run_eegnet_window_sweep.py`
- `scripts/run_eegnet_participant_loo.py`
- `scripts/aggregate_eegnet_window_results.py`
- `scripts/aggregate_participant_loo_results.py`
- `scripts/build_random_epoch_control_dataset.py`
- `docs/STRUKTURA_PROJEKTU.md`
- `.gitignore`
- `postepyprojektu.md`

Weryfikacja:
- Skompilowano wszystkie pliki `*.py` z folderu `scripts/`.
- Sprawdzono `--help` dla `scripts/train_eegnet.py` i `scripts/build_random_epoch_control_dataset.py`.
- Smoke trening z permutacja etykiet: `python .\scripts\train_eegnet.py --dataset-dir event_epoch_window_grid_fine\image_on_0_0p8 --output-dir eegnet_label_permute_smoke --split series --epochs 1 --batch-size 512 --label-control permute --cpu`.
- Smoke permutacji przeszedl: train/validation/test = `1100/440/440`, final test accuracy = `0.1091`; katalog smoke usunieto po sprawdzeniu.
- Smoke datasetu losowych okien: `python .\scripts\build_random_epoch_control_dataset.py --reference-dataset-dir event_epoch_multisession_image_on_0_0p8 --output-dir event_epoch_random_control_smoke --max-rows 2`.
- Smoke losowych okien zapisal `2` epoki kontrolne dla `abc`; katalog smoke usunieto po sprawdzeniu.

## 2026-06-12

### Pelne eksperymenty kontrolne EEGNet

- Uruchomiono pelny leave-one-participant-out z nowym wyborem checkpointu po walidacji:
  `python .\scripts\run_eegnet_participant_loo.py --dataset-dir event_epoch_multisession_image_on_0_0p8 --results-parent eegnet_multisession_results_validated --epochs 12 --batch-size 256 --force`.
- Wynik LOO po walidacji: mean test accuracy `17.19%`, mean validation accuracy `16.84%`.
- Uruchomiono pelny leave-one-participant-out z permutacja etykiet:
  `python .\scripts\run_eegnet_participant_loo.py --dataset-dir event_epoch_multisession_image_on_0_0p8 --results-parent eegnet_multisession_results_permuted --epochs 12 --batch-size 256 --label-control permute --force`.
- Wynik permutacji: mean test accuracy `8.93%`, czyli poziom losowy dla 11 klas.
- Zbudowano pelny dataset kontrolny losowych okien:
  `python .\scripts\build_random_epoch_control_dataset.py --reference-dataset-dir event_epoch_multisession_image_on_0_0p8 --output-dir event_epoch_random_control`.
- Dataset `event_epoch_random_control` zawiera `10120` epok: Bear `2200`, Reshi `1980`, abc `1320`, fghx `2640`, mole `1980`.
- Uruchomiono LOO na losowych oknach:
  `python .\scripts\run_eegnet_participant_loo.py --dataset-dir event_epoch_random_control --results-parent eegnet_random_control_results --epochs 12 --batch-size 256 --force`.
- Wynik losowych okien: mean test accuracy `9.20%`, czyli poziom losowy.
- Uruchomiono split po niewidzianych obrazach:
  `python .\scripts\train_eegnet.py --dataset-dir event_epoch_multisession_image_on_0_0p8 --output-dir eegnet_image_split_results --split image --epochs 12 --batch-size 256`.
- Wynik splitu po `image_id`: test accuracy `20.80%`, validation accuracy `23.15%`.
- Dodano zbiorczy plik wynikow `docs/WYNIKI_EKSPERYMENTOW.md` i wpis w `docs/STRUKTURA_PROJEKTU.md`.

Wniosek:
- Kontrole negatywne sa blisko losowych `9.09%`, a normalny LOO i split po `image_id` zostaja powyzej losowego poziomu.
- To wzmacnia hipoteze, ze model lapie sygnal zwiazany z eventami/klasami, a nie tylko przeciek etykiet, dryft sesji albo przypadkowy kontekst czasowy.

### Aktualizacja dziennika projektu

- Uzupelniono historyczny log `docs/LOG_PROJEKTU.md` o aktualny stan projektu po przejsciu na dataset zdarzeniowy, EEGNet, reorganizacji repozytorium i pelnych kontrolach metodologicznych.
- Wpis obejmuje wyniki: LOO z walidacja `17.19%`, permutacja etykiet `8.93%`, losowe okna `9.20%`, split po `image_id` `20.80%`.
- W logu dodano odniesienie do `docs/WYNIKI_EKSPERYMENTOW.md` oraz `postepyprojektu.md`.
- Dodano aktualne planowane kroki i oznaczono dawny plan VQ-VAE jako historyczny.

Weryfikacja:
- Sprawdzono zawartosc `docs/LOG_PROJEKTU.md` po aktualizacji.
