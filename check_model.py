import xgboost as xgb
import sys

model = xgb.XGBRegressor()
model.load_model('/Volumes/SANDISKUSBC/MLOps/hdb_price_prediction_mlops/streamlit_app/model.xgb')
print("Model features:", model.feature_names_in_)
