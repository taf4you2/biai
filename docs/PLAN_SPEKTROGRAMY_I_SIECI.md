# Plan: spektrogramy, triggery i sieci EEG

## 1. Najpierw dataset zdarzeniowy

Obecny `process_eeg.py` generuje obrazy PSD z rownych 2-sekundowych okien. To jest dobre do eksperymentow z VQ-VAE/anomaliami, ale nie wystarcza do proby odtworzenia obrazu, bo nie ma powiazania z konkretnym bodzcem.

Dodany skrypt `scripts/build_event_spectrogram_dataset.py` tworzy drugi tor danych:

```text
EDF + annotations + events CSV -> epoki wokol triggera -> tensor spektrogramu -> metadata.csv
```

Domyslnie mapowany jest trigger `12 = IMAGE_ON`, czyli moment pokazania zdjecia. Kazda probka ma:

- tensor `.npz` z polem `spectrogram` o ksztalcie `channels x freqs x times`,
- liste kanalow, czestotliwosci i osi czasu,
- metadane: `trial_id`, `image_id`, `image_file`, `image_category`, `event_name`.

Uruchomienie testowe:

```powershell
python .\scripts\build_event_spectrogram_dataset.py --max-trials 3 --output-dir event_spectrogram_dataset_smoke
```

Pelny dataset:

```powershell
python .\scripts\build_event_spectrogram_dataset.py --output-dir event_spectrogram_dataset
```

## 2. Co dac do sieci

Pierwszy wariant wejscia powinien byc prosty i mierzalny:

```text
X = spektrogram: channels x freqs x times
y = image_category albo image_id
```

Na start lepsze jest `image_category` niz `image_id`, bo kategorii jest mniej i po 20 powtorzen na kategorie w serii. Odtwarzanie konkretnego zdjecia jest duzo trudniejsze i wymaga wiecej danych albo modelu z warstwa kontrastowa / embeddingami obrazow.

Drugi wariant wejscia:

```text
X = raw epoch: channels x samples
y = image_category
```

To jest naturalne dla EEGNet. Spektrogram jest naturalniejszy dla malego CNN/ResNet albo U-Net-like encoder.

## 3. Okna czasowe: chwila przed i po

Domyslne okno w skrypcie:

```text
tmin = -0.5 s
tmax =  1.5 s
```

To daje aktywnosc przed bodzcem, reakcje na obraz i krotki slad po obrazie. Do porownania warto wygenerowac trzy datasety:

```powershell
python .\scripts\build_event_spectrogram_dataset.py --tmin -0.5 --tmax 1.5 --output-dir event_spectrogram_dataset_image_on_wide
python .\scripts\build_event_spectrogram_dataset.py --tmin 0.0 --tmax 0.8 --output-dir event_spectrogram_dataset_image_on_early
python .\scripts\build_event_spectrogram_dataset.py --event-code 14 --tmin 0.0 --tmax 2.0 --output-dir event_spectrogram_dataset_describe
```

Interpretacja:

- `IMAGE_ON` sprawdza reakcje na zobaczenie obrazu.
- `DESCRIBE_SCREEN` moze lepiej lapac przypominanie/opisywanie.
- `SPACE_PRESSED` moze sluzyc do analizy konca procesu opisu, ale jest mniej czystym bodzcem.

## 4. Kolejnosc modeli

Proponowana kolejnosc:

1. Baseline cech recznych: band power delta/theta/alpha/beta/gamma per kanal i klasyfikator `LogisticRegression` lub `RandomForest`.
2. EEGNet w PyTorch na surowych epokach `channels x samples`.
3. Maly CNN na spektrogramach `channels x freqs x times`.
4. Dopiero potem U-Net/VQ-VAE, jesli chcemy rekonstrukcje albo embedding, a nie tylko klasyfikacje.

U-Net ma sens, gdy wyjscie tez jest obrazem lub mapa cech. Do klasyfikacji kategorii zdjecia wystarczy encoder CNN bez dekodera. VQ-VAE ma sens jako etap uczenia reprezentacji, ale najpierw trzeba miec baseline, zeby wiedziec, czy sygnal niesie informacje.

## 5. Minimalny eksperyment w Colab

Do Colab najlepiej wrzucic:

- `event_spectrogram_dataset/metadata.csv`,
- folder `event_spectrogram_dataset/tensors`,
- skrypt treningowy PyTorch,
- opcjonalnie miniatury `previews` tylko do kontroli wizualnej.

GPU warto wypozyczyc dopiero po tym, jak lokalnie przejdzie maly test na `--max-trials 50`. Na poczatku bottleneckiem bedzie jakosc mapowania i etykiet, nie moc karty.

## 6. Wyniki lokalnych baseline

Podzial testowy byl grupowany po `series_id`, czyli test zawieral serie niewidziane w treningu.

```text
Losowy baseline:             9.09%
LogisticRegression PSD/STFT: 22.73%
CNN na spektrogramach:       20.45%
EEGNet na raw epokach:       26.36%
```

Najlepszy aktualnie jest EEGNet na surowych epokach `1 x 21 x 1200`.

Nastepny krok: porownac okna czasowe i triggery:

```powershell
python .\scripts\build_event_epoch_dataset.py --tmin 0.0 --tmax 0.8 --output-dir event_epoch_dataset_image_on_early
python .\scripts\build_event_epoch_dataset.py --event-code 14 --tmin 0.0 --tmax 2.0 --output-dir event_epoch_dataset_describe
```

Jesli EEGNet dalej wygrywa, wtedy warto przeniesc eksperyment do Colab/GPU i zrobic strojenie hiperparametrow.

## 7. Porownanie 5 okien czasowych dla EEGNet

Porownano piec okien wokol triggera `12 = IMAGE_ON` na sesji `mole` z `dane/Wyniki`.
Podzial testowy byl taki sam jak wczesniej: grupowanie po `series_id`, test na seriach `[2, 8]`.

```text
image_on_0_0p8     0.0 .. 0.8 s    30.45%
image_on_m02_1p0  -0.2 .. 1.0 s    28.86%
image_on_0_0p5     0.0 .. 0.5 s    27.05%
image_on_m05_1p5  -0.5 .. 1.5 s    26.36%
image_on_pre_m08_0 -0.8 .. 0.0 s   16.14%
```

Najlepsze aktualnie jest okno `IMAGE_ON 0.0 .. 0.8 s`.

Wykryto tez dodatkowe dlugie sesje w `dane/Wyniki`:

```text
abc   1320 IMAGE_ON
Bear  2200 IMAGE_ON
fghx  2640 IMAGE_ON
mole  1980 IMAGE_ON
Reshi 1980 IMAGE_ON
```

Nastepny krok: zbudowac dataset wielosesyjny dla najlepszego okna `0.0 .. 0.8 s`
i sprawdzic EEGNet z testem po uczestniku/sesji.

## 8. Fine sweep okien wokol najlepszego zakresu

Zrobiono wiekszy sweep 8 okien wokol `IMAGE_ON`.
Model i podzial jak wczesniej: EEGNet, test po `series_id`, serie `[2, 8]`.

```text
image_on_0_0p8    0.0 .. 0.8 s    30.45%
image_on_0p1_0p9  0.1 .. 0.9 s    30.45%
image_on_0_0p7    0.0 .. 0.7 s    29.77%
image_on_0_1p0    0.0 .. 1.0 s    29.32%
image_on_0p2_0p8  0.2 .. 0.8 s    29.32%
image_on_0p1_0p8  0.1 .. 0.8 s    28.64%
image_on_0_0p6    0.0 .. 0.6 s    28.18%
image_on_0_0p9    0.0 .. 0.9 s    26.14%
```

Wniosek: dalsze strojenie okien na jednej sesji raczej nie ma juz duzego sensu.
Najlepszy praktyczny wybor to nadal `IMAGE_ON 0.0 .. 0.8 s`, bo jest prostszy
interpretacyjnie i remisuje z `0.1 .. 0.9 s`.

Nastepny etap: uzyc okna `0.0 .. 0.8 s` na wszystkich wykrytych sesjach:
`abc`, `Bear`, `fghx`, `mole`, `Reshi`, a potem testowac generalizacje po uczestniku.

## 9. Dataset wielosesyjny i pierwszy test po uczestniku

Uzyto najlepszego okna `IMAGE_ON 0.0 .. 0.8 s` na wszystkich dlugich sesjach z
`dane/Wyniki`.

```text
abc    1320 epok
Bear   2200 epok
fghx   2640 epok
mole   1980 epok
Reshi  1980 epok
Razem 10120 epok
```

Utworzony dataset:

```text
event_epoch_multisession_image_on_0_0p8
```

Pierwszy test leave-one-participant:

```text
Train: abc, Bear, fghx, Reshi
Test:  mole
EEGNet accuracy: 21.67%
```

Porownanie:

```text
Losowo:               9.09%
Jedna sesja mole:    30.45%
Miedzy uczestnikami: 21.67%
```

Wniosek: sygnal generalizuje ponad losowy poziom, ale roznice osobnicze/sesyjne sa duze.
Nastepny krok: uruchomic leave-one-participant dla pozostalych osob i policzyc srednia.

## 10. Pelny leave-one-participant

Uruchomiono EEGNet dla kazdego testowanego uczestnika na datasiecie:

```text
event_epoch_multisession_image_on_0_0p8
```

Wyniki:

```text
mole   21.67%
Bear   19.77%
abc    19.02%
fghx   17.42%
Reshi  16.26%
MEAN   18.83%
```

Porownanie:

```text
Losowo:                      9.09%
Najlepsze okno jedna sesja: 30.45%
Leave-one-participant mean: 18.83%
```

Wniosek: klasyfikacja generalizuje ponad losowy poziom, ale problemem nr 1 sa
roznice miedzy uczestnikami/sesjami. Kolejne prace powinny isc w:

1. normalizacje per uczestnik / per sesja,
2. rownowazenie liczby probek na uczestnika,
3. trening z walidacja leave-one-participant jako glowna metryka,
4. ewentualnie kalibracje modelu na kilku probkach nowego uczestnika.

## 11. Normalizacja per uczestnik i balansowanie

Dodano do `scripts/train_eegnet.py`:

```text
--normalization global|participant|epoch
--balanced-sampler none|category|participant|category_participant
```

Przetestowano wariant:

```text
--normalization participant
--balanced-sampler category_participant
```

Wyniki LOO:

```text
          global   participant+balanced
mole      21.67%   21.57%
Bear      19.77%   19.05%
abc       19.02%   21.06%
fghx      17.42%   18.48%
Reshi     16.26%   17.42%
MEAN      18.83%   19.52%
```

Wniosek: poprawa jest mala, ale idzie w dobra strone. Normalizacja per uczestnik
i balansowanie pomagaja szczegolnie trudniejszym uczestnikom, ale nie rozwiazuja jeszcze
problemu generalizacji miedzy osobami.
