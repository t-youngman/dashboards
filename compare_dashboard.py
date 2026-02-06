"""
H5 Data Analysis Dashboard

This script creates a Streamlit dashboard to visualize H5 output data from the model.
It should be run from a directory containing H5 output files.

Requirements:
- streamlit
- pandas
- h5py
- plotly
- numpy

Run with (from inet-macro-dev directory):
    streamlit run path/to/macromodel/util/dash_h5.py
"""

import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Set page config
st.set_page_config(
    page_title="Macromodel Dashboard - H5 output",
    page_icon="🌍",
    layout="wide"
)

# Title
st.title("Macromodel Dashboard - H5 output")

# Find all H5 files in the working directory and its subdirectories
cwd = os.getcwd()
h5_files = list(Path(cwd).glob("**/*.h5"))
if not h5_files:
    st.error("No H5 files found in the working directory")
    st.error("Please ensure you are running this script from a directory containing H5 output files")
    st.stop()

# --------------------------------------------
# Sidebar: file selection(s)
# --------------------------------------------
# (Existing single-file picker kept)
selected_file = st.sidebar.selectbox("Select H5 File (denominator: run1)", h5_files)

# >>> NEW: toggle + second file picker for ratio mode (run2 / run1)
compare_mode = st.sidebar.checkbox("Compare two runs (show ratio run2 / run1)", value=False)  # >>> NEW
selected_file2 = None  # >>> NEW
if compare_mode:  # >>> NEW
    candidates = [p for p in h5_files if p != selected_file]  # >>> NEW
    if not candidates:  # >>> NEW
        st.sidebar.warning("Only one H5 file found — comparison disabled.")  # >>> NEW
        compare_mode = False  # >>> NEW
    else:  # >>> NEW
        selected_file2 = st.sidebar.selectbox("Compare against (numerator: run2)", candidates)  # >>> NEW

# Function to load H5 data
def load_h5_data(file_path):
    with h5py.File(file_path, 'r') as f:
        # Get all datasets
        datasets = {}
        def collect_datasets(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets[name] = obj[:]
        f.visititems(collect_datasets)
    return datasets

# Function to parse dataset names into components
def parse_dataset_name(name):
    parts = name.split('/')
    if len(parts) >= 3:
        country = parts[0]
        agent_market = parts[1]
        variable = '/'.join(parts[2:])
        return country, agent_market, variable
    return None, None, name

# Function to get available options based on selections
def get_available_options(data, selected_country=None, selected_agent_market=None):
    available_agent_markets = set()
    available_variables = set()
    
    for name in data.keys():
        country, agent_market, variable = parse_dataset_name(name)
        if country and (selected_country is None or country == selected_country):
            if agent_market:
                available_agent_markets.add(agent_market)
                if selected_agent_market is None or agent_market == selected_agent_market:
                    if variable:
                        available_variables.add(variable)
    
    return sorted(list(available_agent_markets)), sorted(list(available_variables))

# Function to create fan chart
def create_fan_chart(df, title):
    # Calculate statistics
    mean = df.mean()
    q1 = df.quantile(0.25)  # First quartile
    q3 = df.quantile(0.75)  # Third quartile
    d1 = df.quantile(0.1)   # First decile
    d9 = df.quantile(0.9)   # Ninth decile
    
    # Create time points
    time_points = np.arange(len(df))
    
    # Create the fan chart
    fig = go.Figure()
    
    # Add the mean line
    fig.add_trace(go.Scatter(
        x=time_points,
        y=mean,
        name='Mean',
        line=dict(color='black', width=2)
    ))
    
    # Add deciles (from outer to inner)
    fig.add_trace(go.Scatter(
        x=time_points,
        y=d9,
        fill=None,
        mode='lines',
        line_color='rgba(0,100,80,0.2)',
        name='90th Percentile',
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=time_points,
        y=d1,
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,100,80,0.2)',
        name='10th-90th Percentile'
    ))
    
    # Add quartiles (from outer to inner)
    fig.add_trace(go.Scatter(
        x=time_points,
        y=q3,
        fill=None,
        mode='lines',
        line_color='rgba(0,100,80,0.4)',
        name='75th Percentile',
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=time_points,
        y=q1,
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,100,80,0.4)',
        name='25th-75th Percentile'
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title='Value',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

# --------------------------------------------
# Load data
# --------------------------------------------
try:
    # >>> NEW: load run1 + (optionally) run2
    data1 = load_h5_data(selected_file)  # (was: data = load_h5_data(selected_file))  # >>> NEW
    data2 = load_h5_data(selected_file2) if compare_mode else None  # >>> NEW
    
    # Sidebar for dataset selection
    st.sidebar.header("Variable Selection")
    
    # Get initial lists of all components
    # (Build from run1 — denominator)
    countries = sorted(list(set(parse_dataset_name(name)[0] for name in data1.keys() if parse_dataset_name(name)[0])))
    
    # Create vertically stacked dropdowns in the sidebar
    selected_country = st.sidebar.selectbox("Country", countries)
    
    # Get available agent/markets based on selected country
    available_agent_markets, _ = get_available_options(data1, selected_country)
    
    selected_agent_market = st.sidebar.selectbox("Agent/Market", available_agent_markets)
    
    # Get available variables based on selected country and agent/market
    _, available_variables = get_available_options(data1, selected_country, selected_agent_market)
    
    selected_variable = st.sidebar.selectbox("Variable", available_variables)
    
    # Construct the full dataset name
    selected_dataset = f"{selected_country}/{selected_agent_market}/{selected_variable}"
    
    # Main content
    st.header(f"Variable: {selected_dataset}")
    
    # >>> NEW: ratio mode switch & epsilon for safety
    EPS = 1e-12  # small floor to avoid 0-division / NaNs in ratios  # >>> NEW

    if not compare_mode:  # >>> NEW
        # ---- ORIGINAL BEHAVIOR (single run) ----
        try:
            df = pd.DataFrame(data1[selected_dataset])
            df_t = df.T
            
            # Display data shape
            st.write(f"Shape: {data1[selected_dataset].shape}")
            
            # Create visualizations based on data shape
            if len(data1[selected_dataset].shape) == 2 and data1[selected_dataset].shape[1] > 1000:
                # If second dimension is large, show fan chart
                st.subheader("Descriptive Statistics")
                st.write(df_t.describe())
                fig = create_fan_chart(df_t, f"Fan Chart of {selected_dataset}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Otherwise show time series
                # Only show descriptive statistics if there are multiple observations
                if data1[selected_dataset].shape[1] > 1:
                    st.subheader("Descriptive Statistics")
                    st.write(df_t.describe())
                fig = px.line(df, title=f"Time Series of {selected_dataset}")
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Could not convert dataset to DataFrame: {str(e)}")
            st.write("Raw data shape:", data1[selected_dataset].shape)
            st.write("Raw data sample:", data1[selected_dataset][:5])

    else:  # >>> NEW
        # ---- RATIO MODE (run2 / run1) ----  # >>> NEW
        if selected_dataset not in data2:  # >>> NEW
            st.error("Selected variable not found in run2 file.")  # >>> NEW
            st.stop()  # >>> NEW

        a = np.array(data1[selected_dataset])   # denominator (run1)  # >>> NEW
        b = np.array(data2[selected_dataset])   # numerator (run2)    # >>> NEW

        if a.shape != b.shape:  # >>> NEW
            st.error(f"Shape mismatch: run1 {a.shape} vs run2 {b.shape}")  # >>> NEW
            st.stop()  # >>> NEW

        # Safe ratio (b / a), guarding zeros/NaNs  # >>> NEW
        denom = np.where(np.isfinite(a) & (np.abs(a) > EPS), a, EPS)  # >>> NEW
        ratio = np.nan_to_num(b / denom, nan=1.0, posinf=1.0, neginf=1.0)  # >>> NEW

        df_ratio = pd.DataFrame(ratio)  # >>> NEW
        df_ratio_t = df_ratio.T  # >>> NEW

        st.write(f"Shape: {a.shape}")  # >>> NEW
        st.info("Showing ratio = run2 / run1")  # >>> NEW

        # Choose chart type similar to original logic  # >>> NEW
        if ratio.ndim == 2 and ratio.shape[1] > 1000:  # >>> NEW
            st.subheader("Descriptive Statistics (ratio)")  # >>> NEW
            st.write(df_ratio_t.describe())  # >>> NEW
            fig = create_fan_chart(df_ratio_t, f"Fan Chart (ratio) — {selected_dataset}")  # >>> NEW
            st.plotly_chart(fig, use_container_width=True)  # >>> NEW
        else:  # >>> NEW
            fig = px.line(df_ratio, title=f"Time Series (ratio run2 / run1) — {selected_dataset}")  # >>> NEW
            fig.add_hline(y=1.0, line_dash="dash", line_width=1)  # >>> NEW
            st.plotly_chart(fig, use_container_width=True)  # >>> NEW

        # Quick export of ratio CSV  # >>> NEW
        st.download_button(  # >>> NEW
            "Download ratio as CSV",  # >>> NEW
            df_ratio.to_csv(index=False).encode("utf-8"),  # >>> NEW
            file_name=f"ratio_{selected_country}_{selected_agent_market}_{selected_variable.replace('/','_')}.csv",  # >>> NEW
            mime="text/csv"  # >>> NEW
        )  # >>> NEW

except Exception as e:
    st.error(f"Error loading H5 file: {str(e)}")
