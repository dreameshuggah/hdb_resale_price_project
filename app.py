import streamlit as st
import pandas as pd
import numpy as np
import json
import xgboost as xgb
from scipy.spatial.distance import cdist
import plotly.graph_objects as go
import time

# Ensure pandas index compatibility
pd.Int64Index = pd.Index
pd.Float64Index = pd.Index

st.set_page_config(page_title="HDB Resale Price Prediction", page_icon="🏢", layout="wide", initial_sidebar_state="expanded")

# --- Refined Minimalist & Luxurious CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Playfair+Display:ital,wght@0,400;0,600;1,400&display=swap');

/* Base Theme: Deep Charcoal & Soft Gold */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0B0C10;
    color: #C5C6C7;
}

.main {
    background: #0B0C10;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #121418;
    border-right: 1px solid rgba(212, 175, 55, 0.1);
}

/* Typography Hierarchy */
h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: #E8D3A2 !important; /* Soft, muted gold */
    font-weight: 400;
    letter-spacing: 0.5px;
}

/* Refined Metric Cards (Glassmorphism) */
div[data-testid="stMetric"] {
    background: rgba(30, 32, 38, 0.4);
    border-radius: 12px;
    padding: 24px;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    border-color: rgba(232, 211, 162, 0.3);
}

div[data-testid="stMetricValue"] {
    font-size: 2.8rem !important;
    font-family: 'Inter', sans-serif;
    font-weight: 300;
    color: #FFFFFF !important;
    letter-spacing: -1px;
}

div[data-testid="stMetricLabel"] {
    font-size: 0.95rem !important;
    font-weight: 500;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #8A8D93 !important;
}

/* Primary Action Button */
.stButton>button {
    background-color: #E8D3A2;
    color: #0B0C10 !important;
    border: none;
    border-radius: 6px;
    padding: 0.6rem 1.5rem;
    font-weight: 500;
    letter-spacing: 0.5px;
    transition: all 0.2s ease;
    width: 100%;
}

.stButton>button:hover {
    background-color: #F7E7CE;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(232, 211, 162, 0.2);
}

/* Clean up input fields */
.stSelectbox label {
    font-size: 0.85rem !important;
    color: #8A8D93 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Tabs Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 2rem;
    background-color: transparent;
}
.stTabs [data-baseweb="tab"] {
    padding-top: 1rem;
    padding-bottom: 1rem;
    color: #8A8D93;
}
.stTabs [aria-selected="true"] {
    color: #E8D3A2 !important;
    border-bottom-color: #E8D3A2 !important;
}

hr {
    border-color: rgba(255, 255, 255, 0.05);
}

/* Glassmorphism for Charts and Tables */
div[data-testid="stPlotlyChart"], div[data-testid="stDataFrame"] {
    background: rgba(30, 32, 38, 0.4) !important;
    border-radius: 12px !important;
    padding: 20px 0px !important; /* Minimized horizontal padding for maximum width */
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
    margin-bottom: 2rem !important;
    width: 100% !important;
}

/* Ensure dataframes inside glass containers are transparent where possible */
div[data-testid="stDataFrame"] > div {
    background: transparent !important;
}

/* Maximize overall app width for better chart expansion */
[data-testid="stAppViewBlockContainer"] {
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    max-width: 98% !important;
}
</style>
""", unsafe_allow_html=True)

# --- Data Loading (Cached) ---
@st.cache_resource(show_spinner=False)
def load_model():
    model = xgb.XGBRegressor()
    #model.load_model('model.xgb')
    model.load_model('model.json')
    return model

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv('hold.csv')
    index_df = pd.read_csv('hdb_resale_index.csv')
    return df, index_df

@st.cache_data(show_spinner=False)
def load_maps():
    with open('flat_type_rank_map.json', 'r') as f: flat_type_map = json.load(f)
    with open('region_rank_map.json', 'r') as f: region_map = json.load(f)
    with open('town_rank_map.json', 'r') as f: town_map = json.load(f)
    with open('storey_range_rank_map.json', 'r') as f: storey_map = json.load(f)
    with open('flat_model_rank_map.json', 'r') as f: flat_model_map = json.load(f)
    return flat_type_map, region_map, town_map, storey_map, flat_model_map

def derive_quarter_column(df, date_col='month'):
    dt_col = pd.to_datetime(df[date_col])
    df['quarter'] = dt_col.dt.year.astype(str) + ' ' + dt_col.dt.quarter.astype(str) + 'Q'
    return df

# Initialize components
model = load_model()
df, index_df = load_data()
maps = load_maps()
flat_type_map, region_map, town_map, storey_map, flat_model_map = maps

# --- Main App Header ---
st.title("HDB Resale Price Prediction")
st.markdown("<p style='color: #8A8D93; font-size: 1.1rem; margin-top: -10px;'>Algorithmic property assessment powered by historical market data.</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Sidebar Configuration ---
with st.sidebar:
    #st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/HDB_logo.svg/1200px-HDB_logo.svg.png", width=60) # Optional: generic placeholder logo
    st.markdown("### Property Parameters")
    st.markdown("<span style='font-size: 0.8rem; color: #8A8D93;'>Configure the target asset to generate a price estimation model.</span>", unsafe_allow_html=True)
    st.write("")

    # Location Parameters
    with st.expander("📍 Location Details", expanded=True):
        region_opts = sorted(df['region_ura'].unique())
        region = st.selectbox("URA Region", options=region_opts)
        df_filtered = df[df['region_ura'] == region]
        
        town_opts = sorted(df_filtered['town'].unique())
        town = st.selectbox("Planning Area (Town)", options=town_opts)
        df_filtered = df_filtered[df_filtered['town'] == town]
        
        

    # Architectural Parameters
    with st.expander("🏗️ Architectural Details", expanded=True):
        ft_opts = sorted(df_filtered['flat_type'].unique())
        flat_type = st.selectbox("Asset Type", options=ft_opts)
        df_filtered = df_filtered[df_filtered['flat_type'] == flat_type]
        
        fm_opts = sorted(df_filtered['flat_model'].unique())
        flat_model = st.selectbox("Asset Model", options=fm_opts)
        df_filtered = df_filtered[df_filtered['flat_model'] == flat_model]
        
        area_opts = sorted(df_filtered['floor_area_sqm'].unique())
        floor_area_sqm = st.selectbox("Floor Area (SQM)", options=area_opts)
        df_filtered = df_filtered[np.isclose(df_filtered['floor_area_sqm'], floor_area_sqm)]
        
        sr_opts = sorted(df_filtered['storey_range'].unique())
        storey_range = st.selectbox("Elevation Range", options=sr_opts)
        df_filtered = df_filtered[df_filtered['storey_range'] == storey_range]

    # Tenure & Market Condition
    with st.expander("⏳ Tenure & Market Data", expanded=True):
        dist_opts = sorted(df_filtered['distance_to_cbd'].unique())
        distance_to_cbd = st.selectbox("Proximity to CBD (m)", options=dist_opts, format_func=lambda x: f"{x:,.1f}")
        df_filtered = df_filtered[np.isclose(df_filtered['distance_to_cbd'], distance_to_cbd)]

        lease_opts = sorted(df_filtered['remaining_lease_years'].unique())
        remaining_lease_years = st.selectbox("Remaining Lease (Yrs)", options=lease_opts)
        
        sorted_index_df = index_df.sort_values('quarter', ascending=False)
        index_opts = sorted_index_df['resale_price_index'].tolist()
        index_labels = sorted_index_df.apply(lambda x: f"Index: {x['resale_price_index']} ({x['quarter']})", axis=1).tolist()
        resale_price_index = st.selectbox("Market Price Index", options=index_opts, index=0, format_func=lambda x: index_labels[index_opts.index(x)])

    st.write("")
    predict_button = st.button("Generate Price Estimate", use_container_width=True)


# --- Core Execution Logic ---
if predict_button:
    # Simulating a brief loading state for professional feel
    with st.spinner('Synthesizing market data and generating predictive model...'):
        time.sleep(0.8) # Brief artificial delay makes the computation feel substantial to clients
        
        features = pd.DataFrame([{
            'flat_type_rank': flat_type_map.get(flat_type, 0),
            'region_ura_rank': region_map.get(region, 0),
            'town_rank': town_map.get(town, 0),
            'storey_range_rank': storey_map.get(storey_range, 0),
            'flat_model_rank': flat_model_map.get(flat_model, 0),
            'distance_to_cbd': distance_to_cbd,
            'floor_area_sqm': floor_area_sqm,
            'remaining_lease_years': remaining_lease_years,
            'resale_price_index': resale_price_index
        }])
        
        feature_cols = ['flat_type_rank', 'region_ura_rank', 'town_rank', 'storey_range_rank', 'flat_model_rank', 'distance_to_cbd', 'floor_area_sqm', 'remaining_lease_years', 'resale_price_index']
        features = features[feature_cols]
        predicted_price = model.predict(features)[0]
        
        # Data Enrichment & KNN
        df_enriched = derive_quarter_column(df.copy())
        df_enriched = df_enriched.merge(index_df, on='quarter', how='left')
        
        same_town_df = df_enriched[df_enriched['town'] == town].copy()
        if len(same_town_df) < 5: same_town_df = df_enriched.copy()
        
        cont_cols = ['distance_to_cbd', 'floor_area_sqm', 'remaining_lease_years']
        for col in cont_cols:
            if same_town_df[col].std() > 0:
                same_town_df[col + '_z'] = (same_town_df[col] - same_town_df[col].mean()) / same_town_df[col].std()
            else:
                same_town_df[col + '_z'] = 0
                
        input_z = {}
        for col in cont_cols:
            if same_town_df[col].std() > 0:
                input_z[col + '_z'] = (locals()[col] - same_town_df[col].mean()) / same_town_df[col].std()
            else:
                input_z[col + '_z'] = 0
                
        query_point = np.array([[input_z['distance_to_cbd_z'], input_z['floor_area_sqm_z'], input_z['remaining_lease_years_z']]])
        target_points = same_town_df[[c+'_z' for c in cont_cols]].values
        
        distances = cdist(query_point, target_points)[0]
        idx_closest = np.argsort(distances)[:5]
        nearest_neighbors = same_town_df.iloc[idx_closest].copy()
        
        nn_features = pd.DataFrame({
            'flat_type_rank': nearest_neighbors['flat_type'].map(flat_type_map).fillna(0),
            'region_ura_rank': nearest_neighbors['region_ura'].map(region_map).fillna(0),
            'town_rank': nearest_neighbors['town'].map(town_map).fillna(0),
            'storey_range_rank': nearest_neighbors['storey_range'].map(storey_map).fillna(0),
            'flat_model_rank': nearest_neighbors['flat_model'].map(flat_model_map).fillna(0),
            'distance_to_cbd': nearest_neighbors['distance_to_cbd'],
            'floor_area_sqm': nearest_neighbors['floor_area_sqm'],
            'remaining_lease_years': nearest_neighbors['remaining_lease_years'],
            'resale_price_index': nearest_neighbors['resale_price_index']
        })[feature_cols]
        
        nearest_neighbors['predicted_price'] = model.predict(nn_features)
        avg_neighbor_actual = nearest_neighbors['resale_price'].mean()
        
        nn_errors = abs(nearest_neighbors['resale_price'] - nearest_neighbors['predicted_price']) / nearest_neighbors['resale_price'] * 100
        avg_nn_error = nn_errors.mean()
        confidence_score = max(0, 100 - avg_nn_error)

    # --- Dashboard Rendering ---
    
    # 1. Top Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Estimated Market Value", f"${predicted_price:,.0f}")
    with col2:
        st.metric("Model Confidence"
                , f"{confidence_score:.1f}%"
                , delta=f"± {avg_nn_error:.1f}% Margin"
                , delta_color="inverse"
                , help="Calculated as 100% minus the Average Absolute Percentage Error (AAPE) of the model on the 5 most similar historical transactions (neighbors)."
                )
    with col3:
        st.metric("Local Comp. Average", f"${avg_neighbor_actual:,.0f}")

    
    # 2. Historical Context Alert
    exact_matches = same_town_df[
        (same_town_df['flat_type'] == flat_type) & 
        (same_town_df['storey_range'] == storey_range) & 
        (same_town_df['flat_model'] == flat_model) & 
        (np.isclose(same_town_df['floor_area_sqm'], floor_area_sqm, atol=1.0))
    ]
    
    if len(exact_matches) > 0:
        match_actual = exact_matches['resale_price'].mean()
        st.success(f"**Historical Validation:** Identified {len(exact_matches)} identical architectural matches in our database. The average transacted price for these exact assets was **${match_actual:,.0f}**.", icon="✅")
    else:
        st.info("Unique Configuration: No mathematically exact architectural matches were found in the historical data. Relying on spatial and feature proximity.", icon="ℹ️")

    st.write("")
    
    
    # 3. Data Visualization & Deep Dives (Tabs)
    tab1, tab2 = st.tabs(["📊 Market Comparables", "⚙️ Model Payload"])
    
    with tab1:
        st.markdown("#### Retrospective Model Accuracy")
        st.markdown("<p style='color: #8A8D93; font-size: 0.9rem;'>Comparing actual transaction values against our estimator's performance on local comparables.</p>", unsafe_allow_html=True)
        
        fig = go.Figure()
        x_labels = [f"Comp {i+1} ({row['quarter']})" for i, (_, row) in enumerate(nearest_neighbors.iterrows())]
        
        fig.add_trace(go.Bar(
            x=x_labels, 
            y=nearest_neighbors['resale_price'], 
            name='Actual Transacted Price',
            marker_color='#E8D3A2', 
            hovertemplate="Actual: $%{y:,.0f}<extra></extra>"
        ))
        fig.add_trace(go.Bar(
            x=x_labels, 
            y=nearest_neighbors['predicted_price'], 
            name='Model Retrospective Prediction',
            marker_color='#3A3F47',
            hovertemplate="Predicted: $%{y:,.0f}<extra></extra>"
        ))
        
        fig.update_layout(
            barmode='group',
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", size=13, color="#C5C6C7"),
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)", title="Valuation (SGD)"),
            xaxis=dict(gridcolor="rgba(0,0,0,0)", title="")
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        st.markdown("---")
        st.markdown("#### 🗺️ Geospatial Proximity")
        st.markdown("<p style='color: #8A8D93; font-size: 0.9rem;'>Geographic distribution of the identified comparable assets in the target region.</p>", unsafe_allow_html=True)
        
        # Prepare hover text with requested metrics
        hover_text = []
        for _, row in nearest_neighbors.iterrows():
            text = (
                f"<b>Price:</b> ${row['resale_price']:,.0f}<br>"
                f"<b>Date:</b> {row['month']}<br>"
                f"<b>Type:</b> {row['flat_type']}<br>"
                f"<b>Model:</b> {row['flat_model']}<br>"
                f"<b>Area:</b> {row['floor_area_sqm']} SQM<br>"
                f"<b>Storey:</b> {row['storey_range']}<br>"
                f"<b>Block:</b> {row['blk_no']}<br>"
                f"<b>Road:</b> {row['road_name']}"
            )
            hover_text.append(text)

        map_fig = go.Figure(go.Scattermapbox(
            lat=nearest_neighbors['latitude'],
            lon=nearest_neighbors['longitude'],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=14,
                color='#E8D3A2',
                opacity=0.8
            ),
            text=hover_text,
            hoverinfo='text'
        ))

        map_fig.update_layout(
            mapbox_style="carto-darkmatter",
            mapbox=dict(
                center=dict(lat=nearest_neighbors['latitude'].mean(), lon=nearest_neighbors['longitude'].mean()),
                zoom=14.5
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=500,
            dragmode="zoom"
        )
        st.plotly_chart(map_fig, use_container_width=True, config={'displayModeBar': True, 'scrollZoom': True})

        st.markdown("---")
        st.markdown("#### 📋 Comparable Assets Data Table")
        cols_to_show = ['quarter', 'town','road_name','blk_no', 'flat_type','flat_model', 'floor_area_sqm', 'distance_to_cbd', 'remaining_lease_years', 'resale_price', 'predicted_price']
        
        # Clean up column names for client viewing
        display_df = nearest_neighbors[cols_to_show].copy()
        display_df.columns = ['Quarter', 'Town', 'Road Name', 'Block', 'Type', 'Model', 'Area (SQM)', 'Dist. to CBD (m)', 'Lease (Yrs)', 'Actual Price', 'Predicted Price']
        
        st.dataframe(display_df.style.format({
            'Actual Price': '${:,.0f}',
            'Predicted Price': '${:,.0f}',
            'Dist. to CBD (m)': '{:,.1f}',
        }), use_container_width=True, hide_index=True)


    with tab2:
        st.markdown("#### Inference Payload")
        st.markdown("<p style='color: #8A8D93; font-size: 0.9rem;'>Raw engineered feature array passed to the XGBoost estimator.</p>", unsafe_allow_html=True)
        st.json(features.to_dict(orient='records')[0])

else:
    # Empty state instructions
    st.info("👈 Please configure the property parameters in the sidebar and click **Generate Valuation** to begin the assessment.", icon="💡")

# --- Appendix / Data Sources ---
st.markdown("---")
st.markdown("### 📚 Appendix")
st.markdown("""
Assessing the valuation using official historical records and market indices:

1. **HDB Resale Prices**: [Kaggle Dataset](https://www.kaggle.com/datasets/lzytim/hdb-resale-prices)
2. **HDB Resale Price Index (1Q2009 = 100)**: [SingStat Table Builder](https://tablebuilder.singstat.gov.sg/table/TS/M212161)

*Disclaimer: This valuation is an algorithmic estimate based on historical trends and should not replace professional appraisal.*
""")
