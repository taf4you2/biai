# Detaliczny Pipeline Przetwarzania Sygnałów EEG (BIAI)

Ten dokument opisuje kompletny, profesjonalny przepływ danych (pipeline) przygotowujący surowe nagrania z systemu **DSI-24** do modelowania w architekturze **AtomVQVAE**.

---

## Graficzny Schemat Przepływu
`DANE SUROWE` -> `CZYSZCZENIE TECHNICZNE` -> `KONDYCJONOWANIE` -> `DEKOMPOZYCJA ICA` -> **`SMOOTHING (Savitzky-Golay)`** -> `SEGMENTACJA` -> **`SMOOTHING (Gauss)`** -> `AI MODEL`

---

## 1. Akwizycja i Sanityzacja (Faza Wstępna)
*   **Wczytywanie (MNE-Python):** Dane EDF + metadane z CSV.
*   **Rejekcja Impedancyjna:** Kanały oznaczone jako `Low_SNR` w CSV (ponad 50% czasu) są automatycznie wykluczane.
*   **Standaryzacja 10-20:** Mapowanie nazw na czysty standard (np. usuwanie sufiksów "-Ref").

## 2. Kondycjonowanie Fizyczne (Faza DSP)
*   **Filtrowanie 1-40 Hz:** Usuwa dryft linii bazowej i szum mięśniowy (EMG) wysokiej częstotliwości.
*   **Notch Filter (50 Hz):** Wycinanie zakłóceń z sieci elektrycznej.
*   **Common Average Reference (CAR):** Kluczowy krok dla modeli AI. Przelicza potencjał każdej elektrody względem średniej wszystkich innych, eliminując "martwe punkty" fizycznej referencji.

## 3. Zaawansowana Korekcja Artefaktów (Faza ICA)
*   **ICA (Dekompozycja na 15 składowych):** Wyodrębnienie niezależnych źródeł sygnału.
*   **Automatyczna korekcja EOG:** Identyfikacja składowych korelujących z mruganiem (na bazie kanałów czołowych Fp1/Fp2) i ich usunięcie.

## 4. Finalny Szlif: Podwójne Wygładzanie (Advanced Smoothing)
W tym momencie, po usunięciu mrugnięć, stosujemy "niemiecki szlif" dla maksymalnej czytelności:

### A. Wygładzanie Sygnału (Filtr Savitzky-Golay)
*   **Gdzie:** Na oczyszczonym sygnale czasowym (zaraz po ICA).
*   **Zasada:** Dopasowywanie wielomianów niskiego stopnia w ruchomym oknie. 
*   **Cel:** Usunięcie drobnych oscylacji (szumów kwantyzacji), których filtry FIR/IIR nie wycięły, przy jednoczesnym **zachowaniu ostrych krawędzi i fazy** istotnych impulsów EEG.

### B. Wygładzanie Obrazu (Filtr Gaussa)
*   **Gdzie:** Na gotowym spektrogramie (obrazie 128x128).
*   **Zasada:** Konwolucja obrazu z jądrem Gaussa.
*   **Cel:** Redukcja "ziarnistości" spektrogramu. Pomaga to modelowi VQ-VAE skupić się na globalnych strukturach (motywach neuronalnych) zamiast na pojedynczych pikselach szumu.

## 5. Reprezentacja pod VQ-VAE
*   **Segmentacja (Epoching):** Podział na 2-sekundowe okna.
*   **Metoda Welcha:** Obliczanie PSD (Power Spectral Density) dla stabilnych reprezentacji.
*   **Normalizacja Z-Score:** Standaryzacja amplitudy, aby model AI uczył się wzorców, a nie bezwzględnej siły sygnału.
