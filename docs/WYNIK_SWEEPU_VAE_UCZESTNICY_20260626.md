# Sweep rekonstrukcji VAE po uczestnikach — 2026-06-26

## Cel

Domkniecie rekonstrukcji dla wszystkich uczestnikow wariantu `no_abc`, bez
ponownego trenowania modeli EEG -> embedding. Dla `Bear`, `fghx` i `Reshi`
wykorzystano istniejace checkpointy retrieval ResNet-18 oraz DINOv2 i te same
adaptery Ridge -> latent VAE co w poprzednim eksperymencie `mole`.

Kazdy wynik jest liczony na 44 niewidzianych obrazach held-out uczestnika. Wagi
ensemble wybrano wczesniej na walidacji retrieval po usrednieniu powtorzen;
test nie byl uzywany do ich doboru.

## Wyniki EEG -> embedding -> Ridge -> pretrained VAE

| Uczestnik | Waga ResNet / DINO | ResNet SSIM | DINOv2 SSIM | Ensemble SSIM | Ensemble PSNR | Ensemble L1 |
|---|---:|---:|---:|---:|---:|---:|
| Bear | 0.10 / 0.90 | 0.2858 | 0.2887 | **0.2895** | 11.87 dB | 0.2292 |
| fghx | 0.45 / 0.55 | 0.2616 | 0.2789 | **0.2766** | 11.39 dB | 0.2446 |
| Reshi | 0.10 / 0.90 | 0.2691 | 0.2801 | **0.2806** | 11.48 dB | 0.2441 |
| mole | 0.45 / 0.55 | 0.2805 | 0.2858 | **0.2865** | 11.60 dB | 0.2356 |
| Średnia | 0.275 / 0.725 | 0.2742 | 0.2834 | **0.2833** | 11.58 dB | 0.2384 |
| Odch. std. | — | 0.0110 | 0.0047 | 0.0058 | 0.21 dB | 0.0074 |

## Wnioski

- DINOv2 jest lepszy od ResNetu pod wzgledem SSIM dla wszystkich czterech
  uczestnikow.
- Ensemble ma najwyzszy SSIM dla `Bear`, `Reshi` i `mole`; dla `fghx` sam
  DINOv2 jest minimalnie lepszy (`0.2789` vs `0.2766`).
- Wyniki sa stabilne miedzy uczestnikami: ensemble SSIM od `0.2766` do
  `0.2895`.
- Nastepny etap nie powinien polegac na dalszym dostrajaniu klasyfikatora
  kategorii. Ograniczeniem jest mapowanie embeddingu obrazu do latentow VAE,
  dlatego warto porownac obecny Ridge z pretrained image-conditioned
  generatorem.

## Artefakty

- `vae_participant_sweep_no_abc_20260626/`
- `vae_participant_sweep_no_abc_20260626/participant_vae_generation_all_summary.csv`
- `vae_participant_sweep_no_abc_20260626/participant_vae_generation_all_summary.json`
- `scripts/run_no_abc_vae_participant_sweep.ps1`
- `scripts/aggregate_vae_generation_sweep.py`
