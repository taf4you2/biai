# Trening EEGNet w Google Colab

Ten wariant używa gotowego datasetu QC:
`event_epoch_multisession_image_on_0_0p8_qc`.

Nie trzeba przesyłać surowego katalogu `dane/` ani obrazów. Kod jest pobierany
z GitHuba, paczka danych z Google Drive, a trening odbywa się na szybkim dysku
lokalnym Colaba w `/content`.

## Pliki przygotowane lokalnie

Po uruchomieniu:

```powershell
.\scripts\prepare_colab_bundle.ps1
```

powstają:

```text
colab_export/biai_eeg_qc_0_0p8.zip
colab_export/biai_eeg_qc_0_0p8.manifest.json
```

## Jednorazowe udostępnienie

1. Pliki obsługujące Colab są już lokalnie zatwierdzone. Wypchnij gałąź:

```powershell
git push -u origin codex/eeg-pipeline-qc
```

2. Na Google Drive utwórz folder `MyDrive/biai/data`.
3. Prześlij do niego:
   `colab_export/biai_eeg_qc_0_0p8.zip`.
4. Po wykonaniu `git push` otwórz notebook bezpośrednio:
   `https://colab.research.google.com/github/taf4you2/biai/blob/codex/eeg-pipeline-qc/notebooks/TRENING_EEGNET_COLAB.ipynb`.

## Uruchomienie

1. W Colabie wybierz `Środowisko wykonawcze -> Zmień typ środowiska
   wykonawczego -> GPU`.
2. Zamontuj Google Drive i zaakceptuj dostęp.
3. Ustaw parametry w komórce konfiguracyjnej.
4. Uruchom komórkę `TRYB BEZOBSŁUGOWY` i zostaw ją pracującą.

Notebook:

- klonuje gałąź `codex/eeg-pipeline-qc`;
- sam znajduje ZIP, również gdy Drive dopisał do nazwy `(1)`;
- kopiuje ZIP z Drive do `/content`;
- rozpakowuje i weryfikuje wszystkie ścieżki epok;
- sprawdza dostępność GPU;
- wykonuje pełny trening ze splitem `participant_image` i walidacją po
  uczestniku;
- po każdej epoce zapisuje checkpoint, częściową historię i log na Drive;
- po przerwaniu sesji automatycznie wznawia trening;
- zapisuje model, raporty i wykresy na Drive.

## Zmiana eksperymentu

Najważniejsze parametry są w komórce konfiguracyjnej notebooka:

```python
TEST_PARTICIPANT = "mole"
FULL_EPOCHS = 30
BATCH_SIZE = 256
```

Dostępni uczestnicy: `abc`, `Bear`, `fghx`, `mole`, `Reshi`.

Przy błędzie braku pamięci GPU zmniejsz `BATCH_SIZE` do `128` lub `64`.

Google Colab nadal może zakończyć sesję z powodu limitu czasu. W takim
przypadku uruchom ponownie trzy komórki notebooka. Opcja `--resume` odczyta
`training_checkpoint.pt` z Drive i rozpocznie od następnej epoki.

Nie używaj skryptów typu „keep alive” do obchodzenia limitów Colaba. Checkpoint
po każdej epoce zabezpiecza wykonaną pracę w dozwolony sposób.
