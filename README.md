### FPV Analog Video Scanner (HackRF + SoapySDR)

Scanner alle 40 FPV-kanaler (bånd A, B, E, F, R) på 5.8 GHz for analoge videosignaler ved å bruke HackRF via SoapySDR. Detekterer kanal når det finnes en sterk topp minst +8 dB over medianstøyen og et bredbåndig spektrum (≥ 6 MHz).

### Funksjoner
- Skanner alle 40 kanaler: A, B, E, F og R
- FFT-basert deteksjon av analog video
- Tydelig terminal-output per kanal og en samlet oppsummering
- Enkle parametre å justere via kommandolinjen

### Krav
- Python 3.8+
- HackRF One (eller kompatibel)
- SoapySDR og NumPy

Installer Python-avhengigheter:

```bash
pip install SoapySDR numpy
```

Merk (Windows): På Windows finnes det ofte ikke en ferdig `SoapySDR`-wheel på PyPI for din Python-versjon/arkitektur, og `pip install SoapySDR` kan feile. Bruk en av metodene under.

#### Windows-alternativ A (ANBEFALT): conda-forge
1) Installer Miniconda/Anaconda (hvis du ikke har det fra før).
2) Opprett og aktiver et nytt miljø:
```bash
conda create -n fpv python=3.11 -y
conda activate fpv
```
3) Installer SoapySDR, Soapy HackRF og NumPy fra conda-forge:
```bash
conda install -c conda-forge soapysdr soapyhackrf numpy -y
```
4) Verifiser installasjon:
```bash
SoapySDRUtil --info | cat
python -c "import SoapySDR; import numpy; print('SoapySDR OK')"
```

#### Windows-alternativ B: PothosSDR-installer + Python-binding
1) Installer PothosSDR for Windows (inkluderer SoapySDR og SoapyHackRF). Søk etter "PothosSDR Windows" og last ned nyeste installer fra offisielt nettsted.
2) Kontroller at `SoapySDRUtil --info` fungerer i en ny terminal.
3) Installer NumPy og passende `SoapySDR` Python-wheel (last ned wheel som matcher din Python-versjon/bitness fra Pothos/SoapySDR releases), for eksempel:
```bash
pip install numpy
pip install .\SoapySDR-<versjon>-cp311-cp311-win_amd64.whl
```
Hvis du ikke finner riktig wheel, bruk Alternativ A (conda) over.

### Kjapp start
Kjør skanneren fra prosjektmappen:

```bash
python fpv_scanner.py
```

Eksempel-output under skanning:
```
[CLEAR]    Band A - CH1 (5865 MHz)
[DETECTED] Band A - CH3 (5825 MHz)  peak+12.5 dB, bw 6.5 MHz
...

--- RESULTAT ---
Analog FPV video på Band A, Kanal 3 (5825 MHz)
Analog FPV video på Band F, Kanal 7 (5860 MHz)
```

### Parametre
Alle parametre kan endres via kommandolinjen. Standardverdier i parentes.

- `--gain` (30.0): Total RX gain i dB (fordeles mellom LNA og VGA)
- `--sample-rate` (10.0): Sample-rate i MSPS
- `--threshold` (8.0): Terskel over medianstøy i dB for topp-deteksjon
- `--min-bw` (6.0): Minimum båndbredde i MHz for å vurdere analog video
- `--samples` (16384): Antall samples per kanal-måling
- `--settle-ms` (50): Settling-tid etter tuning før måling starter (millisekunder)

Eksempler:

```bash
# Øk gain og prøv lavere terskel
python fpv_scanner.py --gain 35 --threshold 6

# Bruk høyere sample-rate og mer samples per måling
python fpv_scanner.py --sample-rate 12 --samples 32768

# Strengere båndbreddekrav (for å ignorere smalbånd/wifi)
python fpv_scanner.py --min-bw 7.0
```

### Hvordan det virker (kort)
1. Tuner HackRF til hver kanal-frekvens.
2. Leser et blokksett med komplekse samples (CF32).
3. Beregner FFT og effektspektrum i dB.
4. Finner topp over medianstøy og estimerer båndbredde for regionen over terskel.
5. Rapporterer kanal som "DETECTED" hvis både topp og minimum båndbredde oppfylles.

### Test og verifikasjon
1. Koble til HackRF og sørg for at drivere er riktig installert.
2. Start en analog FPV videosender på kjent kanal (f.eks. A3 – 5825 MHz).
3. Kjør `python fpv_scanner.py` og bekreft at kanalen rapporteres som DETECTED i terminalen og i sluttoppsummeringen.

### Feilsøking
- Ingen deteksjoner: Øk `--gain`, senk `--threshold`, eller øk `--samples`.
- Overmetning/for mye støy: Senk `--gain` eller bruk `--threshold` høyere.
- Ingen HackRF funnet: Verifiser driver/installasjon. På Windows, sjekk Zadig/PothosSDR.

### Videre arbeid (idéer)
- Kontinuerlig, rullerende skanning i sanntid
- Logging av signalstyrke over tid (CSV)
- Grafisk spektrumvisning for valgte kanaler
- Automatisk klassifisering av signaltype

### Lisens
MIT



