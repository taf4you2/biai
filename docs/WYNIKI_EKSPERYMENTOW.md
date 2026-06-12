# Wyniki eksperymentow

Ten plik zbiera najwazniejsze wyniki eksperymentow EEGNet i kontroli metodologicznych.

## 2026-06-12: kontrole wiarygodnosci pipeline'u

Dataset bazowy:

```text
event_epoch_multisession_image_on_0_0p8
IMAGE_ON, okno 0.0..0.8 s, 10120 epok, 11 klas
```

Poziom losowy dla 11 klas:

```text
9.09%
```

| Eksperyment | Wynik testowy | Wynik walidacyjny | Interpretacja |
|---|---:|---:|---|
| LOO po uczestniku, checkpoint wybierany po walidacji | 17.19% | 16.84% | Sygnal zostaje ponad losowym po usunieciu wyboru checkpointu po tescie. |
| LOO po uczestniku, normalizacja per uczestnik + sampler `category_participant` | 17.76% | 17.51% | Lekka poprawa wzgledem bazowego LOO, ok. +0.58 pp. |
| LOO po uczestniku, permutacja etykiet | 8.93% | 9.76% | Kontrola spada do poziomu losowego. |
| LOO po uczestniku, losowe okna EEG | 9.20% | 9.83% | Losowe fragmenty sygnalu nie niosa uzytecznej informacji o klasie. |
| Split po `image_id` | 20.80% | 23.15% | Model generalizuje ponad losowo rowniez na niewidziane obrazy, choc to nie jest LOO po uczestniku. |

## 2026-06-12: pelny eksperyment po QC epok

Dataset QC:

```text
event_epoch_multisession_image_on_0_0p8_qc
IMAGE_ON, okno 0.0..0.8 s, 9014 epok zaakceptowanych z 10120 kandydatow, 11 klas
```

QC epok:

| Uczestnik | Epoki po QC | Kandydaci | Odrzucone | Odrzucone % |
|---|---:|---:|---:|---:|
| abc | 1159 | 1320 | 161 | 12.20% |
| Bear | 1935 | 2200 | 265 | 12.05% |
| fghx | 2336 | 2640 | 304 | 11.52% |
| mole | 1661 | 1980 | 319 | 16.11% |
| Reshi | 1923 | 1980 | 57 | 2.88% |
| **Razem** | **9014** | **10120** | **1106** | **10.93%** |

Wszystkie odrzucenia byly przez prog `peak-to-peak`; nie bylo epok z `NaN/inf` ani pelnych flatline.

| Eksperyment | Wynik testowy | Wynik walidacyjny | Interpretacja |
|---|---:|---:|---|
| LOO po uczestniku, QC + normalizacja per uczestnik + sampler `category_participant` | 19.13% | 18.86% | Wynik rosnie wzgledem najlepszego wariantu bez QC `17.76%`, wiec sygnal utrzymuje sie po odrzuceniu artefaktowych epok. |
| Split `participant_image`, test `mole`, QC, EEGNet | 20.42% | 17.14% | Mocniejszy test: held-out participant i held-out `image_id` jednoczesnie; wynik nadal wyraznie ponad losowy. |
| Split `participant_image`, test `mole`, QC, baseline bandpower | 11.71% | n/a | Klasyczne pasma mocy sa tylko lekko ponad losowe i duzo slabsze od EEGNet. |

## Wnioski

- Wczesniejsze wyniki nie wygladaja na prosty efekt przecieku etykiet, bo permutacja etykiet daje `8.93%`.
- Wynik nie wyglada tez na sam dryft sesji albo przypadkowy kontekst czasowy, bo losowe okna daja `9.20%`.
- Po poprawnym wyborze checkpointu po walidacji LOO spada z historycznego `18.83%` do `17.19%`, ale nadal jest wyraznie ponad losowy baseline.
- Normalizacja per uczestnik z samplerem `category_participant` podnosi LOO z `17.19%` do `17.76%`, wiec jest lekko lepszym wariantem treningowym, ale poprawa jest niewielka.
- Split po `image_id` daje `20.80%`, wiec model nie opiera sie wylacznie na zapamietaniu konkretnych obrazow z treningu.
- Po QC epok LOO wzrasta do `19.13%`, co wzmacnia interpretacje, ze poprzedni sygnal nie byl prostym efektem artefaktow wysokiej amplitudy.
- Split `participant_image` na QC daje `20.42%` dla `mole`, ale trzeba go powtorzyc dla pozostalych uczestnikow, zanim stanie sie glowna metryka.

## Pliki wynikowe

- `eegnet_multisession_results_validated/participant_loo_summary.csv`
- `eegnet_multisession_results_validated_participant_balanced/participant_loo_summary.csv`
- `eegnet_multisession_results_permuted/participant_loo_summary.csv`
- `eegnet_random_control_results/participant_loo_summary.csv`
- `eegnet_image_split_results/eegnet_summary.json`
- `event_epoch_random_control/metadata.csv`
- `event_epoch_multisession_image_on_0_0p8_qc/epoch_qc_summary.csv`
- `event_epoch_multisession_image_on_0_0p8_qc/session_summary.csv`
- `eegnet_multisession_results_qc/participant_loo_summary.csv`
- `eegnet_participant_image_qc_results/eegnet_summary.json`
- `baseline_epoch_bandpower_qc_results/bandpower_baseline_summary.json`

## Nastepne sensowne kroki

- Powtorzyc split `participant_image` dla kazdego uczestnika, nie tylko dla `mole`.
- Dodac kontrole negatywne dla datasetu QC: permutacja etykiet i losowe okna po takim samym QC.
- Sprawdzic wariant progow QC, np. `150 uV` vs `200 uV`, zeby ocenic stabilnosc wyniku.
