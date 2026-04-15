# HDB Valuation Intelligence - Project Specification & Status

### Core Application Setup
- [x] Create a Streamlit app to predict HDB resale prices.
- [x] Implement a premium, luxurious dark theme UI with custom CSS (Inter & Playfair Display fonts, muted gold accents, frosted glass metrics).
- [x] Configure maximized app width to allow charts and tables to breathe.

### Data & Model Integration
- [x] Load the trained XGBoost regressor (`model.xgb`).
- [x] Load the historical holdout dataset (`hold.csv`) and the resale index data (`hdb_resale_index.csv`).
- [x] Load engineered rank features from JSON maps (`flat_type`, `region`, `town`, `storey_range`, `flat_model`).
- [x] Implement the `derive_quarter_column` function to map dates to the 'YYYY nQ' format and fetch the correct `resale_price_index`.

### User Interface & Input Fields
- [x] Implement a strict hierarchical filtering system within expandable sidebars (`st.expander`):
    - **Location**: `Region` -> `Town`
    - **Architecture**: `Flat Type` -> `Flat Model` -> `Storey Range` -> `Floor Area SQM`
    - **Tenure/Market**: `Distance to CBD` -> `Remaining Lease Years` -> `Resale Price Index`.
- [x] Add detailed tooltip (`help`) explanations for complex metrics like Model Confidence.

### Prediction Engine & Main Metrics
- [x] Process user inputs into the correctly ordered feature array for the XGBoost model.
- [x] Display the predicted **Estimated Market Value**.
- [x] Calculate and display a **Model Confidence** score (100% minus the Average Absolute Percentage Error of the 5 nearest historical neighbors).
- [x] Implement an Exact Match lookup to find identical architectural configurations in the historical data, validating the prediction against exact physical matches.

### Nearest Neighbors & Geospatial Analysis
- [x] Identify the 5 nearest historical transactions within the same town using distance calculation on Z-score normalized continuous features.
- [x] Create an interactive Plotly grouped bar chart comparing the Actual vs. Predicted prices of the 5 neighbors.
- [x] **[NEW]** Implement an interactive geospatial map (`Scattermapbox`) utilizing a dark cartography theme to plot the physical locations of the 5 comparable assets.
- [x] **[NEW]** Enhance map markers with comprehensive hover tooltips containing Price, Date, Type, Model, Area, Storey, Block, and Road Name.

### Transparency & Data Inspection
- [x] Implement a tabbed interface for deeper data exploration.
- [x] Display a cleanly formatted dataframe containing the detailed profiles of the 5 nearest neighbors (formatted currency, hidden index, human-readable columns).
- [x] Display a raw JSON output of the exact feature array fed into the XGBoost model.
- [x] Include a structured Appendix citing official data sources (Kaggle, SingStat).




