# EEG -> CLIP embedding -> Stable UnCLIP

Ten eksperyment zastępuje adapter `Ridge -> latent VAE` zamrożonym,
pretrained generatorem `StableUnCLIPImg2ImgPipeline`. Generator przyjmuje
bezpośrednio CLIP image embeddings, dlatego targetem EEGNetu są embeddingi
wyciągnięte dokładnie z image encodera tego generatora.

## Pipeline

```text
EEG -> EEGNet -> znormalizowany CLIP image embedding
    -> odtworzona skala embeddingu
    -> Stable UnCLIP Img2Img (zamrożony)
    -> obraz
```

Do porównania skrypt generacji tworzy też wariant `oracle`, z prawdziwym
embeddingiem CLIP zamiast embeddingu przewidzianego z EEG. Rozdziela to błąd
EEGNetu od ograniczeń samego generowania.

## Dane do Colaba

Na komputerze uruchom:

```powershell
.\scripts\prepare_unclip_colab_assets.ps1
```

Powstanie `colab_export/biai_unclip_assets.zip`. Prześlij go na prywatny Drive
do `MyDrive/biai/data/` obok istniejącego `biai_eeg_qc_0_0p8.zip`.

Gotowy notebook: `notebooks/REKONSTRUKCJA_UNCLIP_COLAB.ipynb`. Notebook
domyślnie robi smoke test na 2 obrazach, a po nim automatycznie uruchamia pełne
generowanie do osobnego folderu `unclip_mole_generation_full`.

## Kolejność uruchomienia

1. W Colabie użyj GPU.
2. Rozpakuj istniejący ZIP z epokami QC oraz `biai_unclip_assets.zip`.
3. Wyciągnij embeddingi:

```bash
python scripts/extract_unclip_image_embeddings.py \
  --manifest-dir reconstruction_manifests/participant_image_mole_no_abc \
  --project-root . \
  --output-dir image_embeddings_unclip_participant_image_mole_no_abc
```

4. Wytrenuj EEG -> embedding:

```bash
python scripts/train_eeg_image_retrieval.py \
  --manifest-dir reconstruction_manifests/participant_image_mole_no_abc \
  --project-root . \
  --embedding-dir image_embeddings_unclip_participant_image_mole_no_abc \
  --output-dir /content/drive/MyDrive/biai/results/unclip_mole_retrieval \
  --epochs 25 --batch-size 128
```

5. Najpierw wygeneruj mały smoke test (`--max-images 2`), potem pełny wynik.
   Notebook robi oba kroki automatycznie, ale ręcznie odpowiada to:

```bash
python scripts/generate_unclip_from_eeg.py \
  --retrieval-result-dir /content/drive/MyDrive/biai/results/unclip_mole_retrieval \
  --project-root . \
  --output-dir /content/drive/MyDrive/biai/results/unclip_mole_generation \
  --oracle --num-inference-steps 20
```

Model `diffusers/stable-diffusion-2-1-unclip-i2i-l` zostanie pobrany przy
pierwszym użyciu i wymaga GPU. Nie uruchamiaj pełnej generacji lokalnie na CPU.
Trening EEG -> embedding zapisuje checkpoint po każdej epoce; ponowne
uruchomienie z `--resume` kontynuuje od ostatniej ukończonej epoki.

## Analiza wyników

Po pobraniu ZIP-a z Drive uruchom `notebooks/ANALIZA_WYNIKOW_COLAB_REKONSTRUKCJA.ipynb`.
Notebook automatycznie:

1. rozpoznaje, czy wynik to tylko `smoke`, czy pełne `full`,
2. wczytuje metryki EEG -> CLIP retrieval i Stable UnCLIP,
3. porównuje wynik z lokalnym VAE baseline,
4. pokazuje gridy wygenerowanych obrazów,
5. wypisuje decyzję, czy odpalać pełne generowanie, czy przejść do kolejnego
   eksperymentu poprawiającego dekoder/generator.

W Colabie notebook próbuje sam zamontować Google Drive, sprawdza typowe
lokalizacje (`MyDrive/biai/results`, `MyDrive/wyniki colab`, `MyDrive/biai`) i
awaryjnie szuka katalogów/ZIP-ów z wynikami po całym `MyDrive`.

Jeśli potrzebujesz jednego miejsca opisującego cały przepływ pracy w Colabie
oraz logikę interpretacji wyników, użyj
`notebooks/PRZEWODNIK_COLAB_I_ANALIZA_WYNIKOW.ipynb`.

## Następny eksperyment po pełnym UnCLIP

Pełny wynik `unclip_mole_generation_full` dla `mole` dał niższe podobieństwo
niż VAE baseline: UnCLIP z EEG osiągnął około `SSIM=0.173`, oracle około
`SSIM=0.246`, a VAE ensemble dla `mole` około `SSIM=0.286`. To sugeruje, że
pretrained Stable UnCLIP nie jest jeszcze dobrym bezpośrednim generatorem dla
tego sygnału i tych bodźców.

Kolejny notebook to
`notebooks/RETRIEVAL_RERANKING_PO_UNCLIP_COLAB.ipynb`. Sprawdza alternatywny
kierunek: użycie EEG -> CLIP embedding jako mechanizmu retrieval/reranking,
czyli wyboru najbliższego obrazu/kandydata po uśrednieniu powtórzeń, zamiast
bezpośredniego generowania obrazu z rozmytego embeddingu EEG. Notebook porównuje
wynik z VAE oraz z pełnym UnCLIP i zapisuje gridy najlepszych/najgorszych
rekonstrukcji retrieval.

Po wynikach retrieval/reranking wykryliśmy silny problem hubness: 44 obrazy
testowe zapadały się do 3 najczęściej wybieranych kandydatów. Następny notebook
`notebooks/KOREKCJA_HUBNESS_RERANKING_COLAB.ipynb` testuje korekcje scoringu
kandydatów bez ponownego trenowania modelu: centrowanie kandydatów, z-score,
CSLS oraz oracle ograniczony do prawdziwej kategorii.

Lokalny run tego notebooka po wyczerpaniu Colaba wskazał `candidate_zscore`
jako najlepszą metodę bez oracle: `top1=13.64%`, `top5=36.36%`,
`category_top1=29.55%`, `SSIM=0.264` i 24 różne przewidywane obrazy zamiast 3
dla surowego cosinusa. To nadal jest lekko poniżej VAE ensemble dla `mole`
(`SSIM≈0.286`), ale wyraźnie powyżej bezpośredniego Stable UnCLIP EEG
(`SSIM≈0.173`) oraz Stable UnCLIP oracle (`SSIM≈0.246`). Oracle ograniczony do
prawdziwej kategorii osiągnął `SSIM≈0.431`, więc następny etap powinien iść w
stronę category-aware reconstruction: najpierw zawężenie/predykcja kategorii,
potem reranking albo generowanie wewnątrz tej kategorii.
