# Unsupervised Detection of Market Regimes in the Colombian Electricity Spot Market

Undergraduate thesis in Industrial Engineering, Universidad de los Andes (202620).
Author: Daniel Benavides. Advisor: Juan Fernando Pérez.
The thesis document is in [`paper/2026_Unsupervised_Detection_of_Market_Regimes.pdf`](paper/2026_Unsupervised_Detection_of_Market_Regimes.pdf).

## Overview

Daily and weekly windows of the hourly spot price are transformed into scalograms with the continuous wavelet transform (CWT). A variational autoencoder with a recurrent layer (VAE-LSTM) summarizes each day in a latent vector, and k-means groups these vectors, together with the mean and standard deviation of the price, into 3 to 5 regimes. The regimes are compared with baselines that use only the mean and standard deviation of the price.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

- **Data:** run the scripts in order.

  ```bash
  python scripts/01_download_xm_data.py --all
  python scripts/02_process_raw_data.py
  python scripts/03_process_external_data.py
  python scripts/04_generate_scalograms.py --price log                   # daily
  python scripts/04_generate_scalograms.py --price log --window week     # rolling weekly
  python scripts/04_generate_scalograms.py --price log --window calweek  # calendar week
  python scripts/05_el_nino_price_stats.py
  ```

- **Models:** run `notebooks/02_vae_lstm.ipynb`, `03_vae_lstm_weekly.ipynb` and `04_vae_lstm_calweek.ipynb` (VAE-LSTM), `05_level_baseline.ipynb` (baselines) and `06_latent_smoothing.ipynb` (backward 7-day mean). Runs are saved in `experiments/`.
- **Metrics:** run `python scripts/06_regime_metrics.py` and `python scripts/07_calweek_metrics.py`, then `notebooks/07_top_models.ipynb` to select the best configurations.

---

## Project Structure

- `scripts/`: Data pipeline and metrics.
- `src/`: Code imported by the notebooks and scripts.
  - `models/vae_lstm.py`: VAE-LSTM (CNN per day and a causal LSTM over consecutive days).
  - `models/conv_vae.py`: Convolutional encoder and decoder used by the VAE-LSTM.
  - `data.py`, `training.py`, `clustering.py`, `metrics.py`: Data loading, training loop, k-means and metrics.
  - `experiments.py`, `runs.py`, `plots.py`: Run folders, registry and figures.
- `notebooks/`: Exploratory analysis (01), models (02 to 06) and model selection (07).
- `paper/`: The thesis document (PDF).
- `metrics/`: Metrics of every regime assignment, written by scripts 06 and 07.
- `data/`: Raw and processed data from XM, DANE and NOAA. Scalograms (`.npy`) are not in git.

---

## Citation

```bibtex
@thesis{benavides2026,
  author      = {Benavides, Daniel},
  title       = {Unsupervised Detection of Market Regimes in the Colombian Electricity Spot Market with Wavelet Scalograms and Variational Autoencoders},
  type        = {Undergraduate thesis},
  institution = {Universidad de los Andes},
  year        = {2026}
}
```

Contact: <d.benavidess@uniandes.edu.co>
