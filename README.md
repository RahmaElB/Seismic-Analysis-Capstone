# Seismic Data Analysis in Morocco

This capstone project investigates earthquake patterns in Morocco and worldwide using machine learning and deep learning models. It was developed in response to the devastating September 2023 earthquake in Morocco and aims to improve seismic monitoring and preparedness through AI tools and predictive modeling.

---

## Project Goals

- Analyze and visualize global and Moroccan earthquake data.
- Preprocess and clean historical data for robust machine learning input.
- Build and evaluate models (KNN, Random Forest, LSTM) to classify or forecast earthquake magnitudes.
- Provide real-time insights and recommendations based on data patterns.

---

## Datasets

- **Global Dataset**: Sourced from the USGS via Kaggle, spanning 1906 to 2022 with 283,132 seismic records. Due to size, the dataset is hosted externally: https://drive.google.com/file/d/1nCuffIRP_PvQUzrsZqy8tn6-CaBBcKrf/view?usp=sharing 
- **Moroccan Dataset**: Collected in collaboration with the Institut National de Géophysique (ING), spanning 1900 to 2023 with 65,931 records.

---

## Tools and Libraries

- Python (Jupyter Notebooks / Anaconda)
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`, `tensorflow`, `keras`, `geopandas`, `shapely`

---

## Models Used

- **KNN**: Classification of earthquake magnitude ranges using neighborhood-based learning.
- **Random Forest**: Regression model for predicting continuous earthquake magnitudes using ensemble decision trees.
- **LSTM**: Time-series model capturing sequential dependencies in magnitude data using Long Short-Term Memory networks.

---

## Results

- **Random Forest** performed well in predicting continuous magnitude values with lower error and good generalization.
- **LSTM** effectively captured time-dependence in the data, achieving acceptable Root Mean Square Error (RMSE).
- **KNN** was less accurate overall, especially for underrepresented magnitude classes, but served as a baseline classifier.
