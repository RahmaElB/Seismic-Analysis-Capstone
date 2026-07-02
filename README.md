# Seismic Analysis Capstone

Machine learning analysis of Moroccan earthquake patterns (1901–2023), developed as a bachelor thesis in response to the devastating September 2023 Al Haouz earthquake (M6.8). The project applies KNN, Random Forest, and LSTM models to a historical seismic catalog to classify and forecast earthquake magnitudes.

---

## Context

On September 8, 2023, a 6.8 magnitude earthquake struck Al Haouz Province in Morocco, killing over 2,900 people. This project was motivated by that event and uses earthquake data collected in collaboration with the **Institut National de Géophysique (ING)** alongside the global USGS catalog to analyze seismic patterns and evaluate machine learning approaches for magnitude prediction.

---

## Dataset

| Dataset | Source | Records | Period |
|---------|--------|---------|--------|
| Moroccan catalog | Institut National de Géophysique (ING) | 65,578 (after cleaning) | 1901–2023 |
| Global catalog | USGS via Kaggle | 283,132 | 1906–2022 |

The Moroccan dataset (`cleaned_earthquake_data.csv`) is included in this repository. The global dataset is hosted externally due to size: [Google Drive link](https://drive.google.com/file/d/1nCuffIRP_PvQUzrsZqy8tn6-CaBBcKrf/view?usp=sharing).

**Data quality issues identified and corrected:**
- 2 rows with corrupt longitude values (−571, −4,251): removed
- 1 future-dated record (year = 2110): removed
- 338 rows with negative magnitudes (catalog noise): removed
- 2 depth outliers beyond 2,000 km: removed

---

## Project Structure

```
Seismic-Analysis-Capstone/
├── notebooks/
│   ├── 01_data_exploration.ipynb     # EDA, cleaning, spatial/temporal analysis
│   └── 02_model_comparison.ipynb     # KNN, Random Forest, LSTM training & evaluation
├── data/
│   └── cleaned_earthquake_data.csv   # Moroccan dataset (ING)
├── figures/                          # Generated plots (auto-created when notebooks run)
├── requirements.txt
└── README.md
```

---

## Models & Results

### K-Nearest Neighbors: Magnitude Classification

Classifies earthquakes into five magnitude classes: Micro (0–2), Minor (2–3), Light (3–4), Moderate (4–5), Strong (5+).

| Metric | Score |
|--------|-------|
| Test accuracy | **73.6%** |
| Best k | 11 |
| Weighting | distance |

KNN performs well on the dominant classes (Micro: F1 = 0.86) but struggles on rare high-magnitude events, a known consequence of class imbalance (strong events: < 0.2% of records). This reflects the Gutenberg-Richter law, smaller earthquakes are exponentially more frequent.

### Random Forest: Magnitude Regression

Predicts continuous Richter magnitude using spatial and temporal features.

| Metric | Score |
|--------|-------|
| RMSE | **0.717** |
| MAE | **0.489** |
| R² | **0.512** |

Feature importance analysis revealed that `year` (0.333) and geographic coordinates (`longitude` 0.202, `latitude` 0.193) are the strongest predictors — suggesting that long-term catalog improvements and regional tectonic patterns drive predictability more than short-term temporal features.

### LSTM: Time-Series Forecasting

Forecasts daily mean magnitude from 60-day sliding windows of historical observations. Captures temporal autocorrelation in seismic sequences (aftershock decay, seasonal clustering). Early stopping triggered at epoch 8 (best weights from epoch 3).

| Metric | Score |
|--------|-------|
| RMSE | **0.475** |
| MAE | **0.314** |
| Architecture | LSTM(64) → Dropout(0.2) → LSTM(32) → Dense(1) |
| Epochs trained | 8 (early stopping) |

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/RahmaElB/Seismic-Analysis-Capstone.git
cd Seismic-Analysis-Capstone
```

### 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the notebooks

```bash
jupyter notebook
```

Open and run in order:
1. `notebooks/01_data_exploration.ipynb`: EDA and cleaning
2. `notebooks/02_model_comparison.ipynb`: model training and evaluation

Figures are saved to `figures/` automatically.

---

## Key Findings

- **LSTM achieved the lowest error** on the time-series forecasting task (RMSE = 0.475, MAE = 0.314), converging quickly in just 8 epochs with early stopping. This reflects the strong temporal autocorrelation in daily seismic activity, aftershock sequences produce predictable short-term magnitude patterns.

- **Random Forest** is the strongest model for point-in-time magnitude regression (RMSE = 0.717, R² = 0.512). The dominance of `year` as a predictor reflects historical catalog sparsity before ~1980 — modern seismometer networks detect more low-magnitude events, making year a proxy for detection threshold rather than true seismicity increase.

- **Class imbalance** is the primary challenge for KNN (73.6% accuracy). The dataset follows the Gutenberg-Richter law: for every unit increase in magnitude, the number of events decreases by roughly an order of magnitude. Micro-events (0–2) make up 54.3% of the catalog; strong events (5+) only 0.2%. Future work could apply SMOTE oversampling or cost-sensitive learning to improve recall on higher-magnitude classes.

- **94.7% of earthquakes are shallow** (depth < 70 km), consistent with the tectonic setting of Morocco at the convergent boundary between the African and Eurasian plates.

---

## Tools & Libraries

Python · Jupyter · pandas · NumPy · scikit-learn · TensorFlow/Keras · matplotlib · seaborn

---

## Related Work

This project was extended into a full MLOps pipeline in [seismic-mlops](https://github.com/RahmaElB/seismic-mlops), which adds MLflow experiment tracking, a FastAPI serving layer, Docker containerization, and a model registry with versioning.

---

## Acknowledgements

Seismic catalog data provided by the Institut National de Géophysique (ING), Morocco. Global dataset sourced from the USGS Earthquake Hazards Program via Kaggle.
