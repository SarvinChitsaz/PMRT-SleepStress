# Data

## CAP Sleep Database

Public polysomnography collection from the Sleep Disorders Center of the Ospedale Maggiore of Parma (PhysioNet).

- Page: https://physionet.org/content/capslpdb/1.0.0/ (DOI 10.13026/C2VC79)
- License: Open Data Commons Attribution License v1.0
- 108 recordings; this project uses 103 subjects (see below)
- Cite: Terzano MG et al. Atlas, rules, and recording techniques for the scoring of cyclic alternating pattern (CAP) in human sleep. Sleep Med 2001;2(6):537-553, and Goldberger AL et al. PhysioBank, PhysioToolkit, and PhysioNet. Circulation 2000;101(23):e215-e220.

The raw data and the derived arrays are **not** part of this repository.

## Pipeline

1. `notebooks/cap_download.ipynb` downloads the records and checks SHA-256.
2. `notebooks/cap_preprocessing.ipynb` writes one `.npz` per record:
   - EEG, chin EMG (and ECG for labels only) band-pass filtered with a 2nd-order Butterworth filter
     (EEG 0.5-30 Hz, EMG 10-60 Hz, ECG 0.5-30 Hz), resampled to 125 Hz
   - 5-s windows (2 channels x 625 samples)
   - labels per window: heart rate (RR intervals in the 5-s window, at least 3 beats), RMSSD (60-s context), sleep stage
3. `python main.py build-cache --npz-dir <cap_processed> --cache-dir <cap_dataset_cache>` merges the `.npz` files into
   `x.npy` (float16, N x 2 x 625) and `index.csv`.

ECG is used only to build HR and RMSSD labels. The model never receives it.

## Groups

Healthy, NFLE, RBD, PLM, Insomnia, Narcolepsy form the 6-class disorder head. SDB (4 subjects) and Bruxism (2) have too
few subjects for that head; their windows are still used for HR, RMSSD and sleep stage (label ignored for the disorder loss).

## Stress index

`SI = z(HR) - z(RMSSD)`, z-scored within each record on raw RMSSD, stress if `SI > 1` (24.2% of windows).
Equal weights and the threshold 1 are assumptions of this work, not taken from a source.
