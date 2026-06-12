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
| LOO po uczestniku, permutacja etykiet | 8.93% | 9.76% | Kontrola spada do poziomu losowego. |
| LOO po uczestniku, losowe okna EEG | 9.20% | 9.83% | Losowe fragmenty sygnalu nie niosa uzytecznej informacji o klasie. |
| Split po `image_id` | 20.80% | 23.15% | Model generalizuje ponad losowo rowniez na niewidziane obrazy, choc to nie jest LOO po uczestniku. |

## Wnioski

- Wczesniejsze wyniki nie wygladaja na prosty efekt przecieku etykiet, bo permutacja etykiet daje `8.93%`.
- Wynik nie wyglada tez na sam dryft sesji albo przypadkowy kontekst czasowy, bo losowe okna daja `9.20%`.
- Po poprawnym wyborze checkpointu po walidacji LOO spada z historycznego `18.83%` do `17.19%`, ale nadal jest wyraznie ponad losowy baseline.
- Split po `image_id` daje `20.80%`, wiec model nie opiera sie wylacznie na zapamietaniu konkretnych obrazow z treningu.

## Pliki wynikowe

- `eegnet_multisession_results_validated/participant_loo_summary.csv`
- `eegnet_multisession_results_permuted/participant_loo_summary.csv`
- `eegnet_random_control_results/participant_loo_summary.csv`
- `eegnet_image_split_results/eegnet_summary.json`
- `event_epoch_random_control/metadata.csv`

## Nastepne sensowne kroki

- Powtorzyc najlepszy LOO z `--normalization participant --balanced-sampler category_participant` juz w trybie z walidacja.
- Uruchomic split mieszany: held-out participant + held-out `image_id`, jesli dodamy taki tryb splitu.
- Dodac raport QC epok: amplitudy, peak-to-peak, odrzucanie artefaktow i liczba odrzuconych probek per uczestnik.
