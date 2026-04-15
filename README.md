# HDB Resale Price Prediction 
https://hdb-resale-price-prediction-engine.streamlit.app/

A robust, machine-learning-powered web application designed to predict and analyze the resale value of Housing & Development Board (HDB) flats in Singapore. Built with Streamlit and XGBoost, this tool provides algorithmic property assessments backed by historical market data, geospatial proximity analysis, and retrospective model validation.

## 🌟 Key Features

* **Advanced Predictive Engine:** Utilizes an XGBoost regression model trained on comprehensive historical HDB transaction data.
* **Geospatial & Feature Comparables:** Employs K-Nearest Neighbors (KNN) logic on Z-score normalized features (Distance to CBD, Floor Area, Remaining Lease) to identify the 5 most relevant historical market comparables.
* **Confidence Scoring:** Calculates dynamic prediction confidence based on the Average Absolute Percentage Error (AAPE) of the localized comparable assets.
* **Interactive Mapping:** Features a dark-themed Plotly `Scattermapbox` plotting the exact geographical coordinates of comparable properties with rich metadata tooltips.
* **Premium UI/UX:** Designed with a clean, minimalist, and luxurious dark interface utilizing custom CSS, glassmorphism elements, and professional typography.
* **Data Transparency:** Allows users to inspect the raw inference payload, detailed data tables of comparable assets, and algorithmic retrospective accuracy charts.

## 🛠️ Tech Stack

* **Frontend:** Streamlit, Custom CSS
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** XGBoost, SciPy (Spatial Distance)
* **Visualizations:** Plotly Graph Objects

## 🚀 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/dreameshuggah/hdb_resale_price_streamlit.git](https://github.com/dreameshuggah/hdb_resale_price_streamlit.git)
   cd hdb_resale_price_streamlit
