import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
from macro_data import DataWrapper
import math

# Set page config
st.set_page_config(
    page_title="Macromodel Validation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
VARIABLES_OF_INTEREST = {
    'Gross Output': 'Gross Output',
    'CPI': 'CPI',
    'Unemployment Rate': 'Unemployment Rate'
}
CONFIDENCE_LEVELS = {
    '90%': 0.90,
    '10%': 0.10
}

@st.cache_data
def load_historical_data():
    """Load the full exogenous data series for historical data."""
    data = DataWrapper.init_from_pickle("data/processed_data/data.pkl")
    country_data = data.synthetic_countries['GBR']
    exo = country_data.exogenous_data

    df = pd.DataFrame({
        'Gross Output': exo.national_accounts['Gross Output (Value)'],
        'CPI': exo.national_accounts['CPI (Value)'] if 'CPI (Value)' in exo.national_accounts else np.nan,
        'Unemployment Rate': exo.labour_stats['Unemployment Rate (Value)'] if hasattr(exo, 'labour_stats') and 'Unemployment Rate (Value)' in exo.labour_stats else np.nan
    })
    df = df.ffill().bfill()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill()
    # Set the start year to match the first available data point
    if len(df) > 0:
        start_year = df.index[0].year if isinstance(df.index[0], pd.Timestamp) else 2000
    else:
        start_year = 2000
    # If the index is not already a DatetimeIndex, create one
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.date_range(start=f'{start_year}-01-01', periods=len(df), freq='Q')
    # Rebase CPI so that the first simulation period (2014-Q1) is 100
    sim_start = pd.Timestamp('2014-01-01')
    if 'CPI' in df.columns and sim_start in df.index:
        cpi_base = df.loc[sim_start, 'CPI']
        df['CPI'] = df['CPI'] / cpi_base * 100
    return df

@st.cache_data
def load_monte_carlo_results(results_dir):
    """Load and process Monte Carlo simulation results, and use nominal Gross Output levels"""
    results_dir = Path(results_dir).resolve()
    all_runs = []
    for file in sorted(results_dir.glob("UK_run_*.csv")):
        try:
            run_data = pd.read_csv(file, index_col=0)
            # Use nominal Gross Output level
            run_data['Gross Output'] = run_data['Gross Output']
            # Rebase CPI so that the first simulation period is 100
            if 'CPI' in run_data.columns:
                cpi_base = run_data['CPI'].iloc[0]
                run_data['CPI'] = run_data['CPI'] / cpi_base * 100
            # Unemployment Rate: use as is if present, or fallback to 'Unemployment Rate (Value)', else np.nan
            if 'Unemployment Rate' not in run_data.columns:
                if 'Unemployment Rate (Value)' in run_data.columns:
                    run_data['Unemployment Rate'] = run_data['Unemployment Rate (Value)']
                else:
                    run_data['Unemployment Rate'] = np.nan
            all_runs.append(run_data)
        except Exception as e:
            st.warning(f"Error loading {file}: {str(e)}")
            continue
    if not all_runs:
        raise ValueError(f"No simulation results found in {results_dir}")
    combined_runs = pd.concat(all_runs, keys=range(len(all_runs)), names=['run', 'time'])
    return combined_runs

@st.cache_data
def load_var_forecast(results_dir, historical_data=None):
    """Load VAR forecast results for GBR, supporting both wide and long formats. Transforms inflation to index and GDP growth to level if historical_data is provided."""
    try:
        var_file = Path(results_dir) / 'var_output.csv'
        if not var_file.exists():
            return None
        var_data = pd.read_csv(var_file)
        # Handle long format: ['country', 'variable', 'period', 'value']
        if set(['country', 'variable', 'period', 'value']).issubset(var_data.columns):
            # Filter for GBR
            gbr_data = var_data[var_data['country'] == 'GBR']
            # Aggregate by period and variable (mean)
            gbr_agg = gbr_data.groupby(['period', 'variable'], as_index=False)['value'].mean()
            # Pivot so each variable is a column, indexed by period
            pivot = gbr_agg.pivot(index='period', columns='variable', values='value')
            # Map variable names based on actual columns
            var_mapping = {
                'real_gdp_growth': 'Gross Output',
                'inflation': 'CPI',
                # Add unemployment mapping if present, e.g. 'unemployment': 'Unemployment Rate'
            }
            var_forecast = pd.DataFrame()
            for var_col, std_col in var_mapping.items():
                if var_col in pivot.columns:
                    var_forecast[std_col] = pivot[var_col]
            # Set index as quarterly dates if period is integer
            period_values = var_forecast.index.values
            if all(isinstance(p, (int, np.integer)) for p in period_values) or np.issubdtype(var_forecast.index.dtype, np.integer):
                start = pd.Timestamp('2014-01-01')
                var_forecast.index = pd.date_range(start=start, periods=len(var_forecast), freq='Q')
            else:
                def parse_period(p):
                    import re
                    m = re.match(r"(\d{4})Q(\d)", str(p))
                    if m:
                        year, q = int(m.group(1)), int(m.group(2))
                        month = (q - 1) * 3 + 1
                        return pd.Timestamp(year=year, month=month, day=1)
                    return pd.NaT
                var_forecast.index = [parse_period(p) for p in var_forecast.index]
            # Transform CPI (inflation) to index
            if 'CPI' in var_forecast.columns:
                cpi_index = [100]
                for infl in var_forecast['CPI'].iloc[1:]:
                    cpi_index.append(cpi_index[-1] * (1 + infl/100))
                var_forecast['CPI'] = cpi_index[:len(var_forecast)]
            # Transform GDP growth to level using 2013Q4 historical value if available
            if 'Gross Output' in var_forecast.columns and historical_data is not None:
                # Find 2013Q4 value
                base_date = pd.Timestamp('2013-10-01')
                if base_date in historical_data.index:
                    base_val = historical_data.loc[base_date, 'Gross Output']
                else:
                    base_val = historical_data['Gross Output'].iloc[-1]  # fallback
                go_index = [base_val]
                for growth in var_forecast['Gross Output'].iloc[1:]:
                    go_index.append(go_index[-1] * (1 + growth/100))
                var_forecast['Gross Output'] = go_index[:len(var_forecast)]
            return var_forecast
        # Handle wide format (old logic)
        gbr_cols = [col for col in var_data.columns if 'GBR' in col]
        if not gbr_cols:
            return None
        var_mapping = {
            'GBR_GDP': 'Gross Output',
            'GBR_CPI': 'CPI',
            'GBR_UNEMP': 'Unemployment Rate'
        }
        var_forecast = pd.DataFrame()
        for var_col, std_col in var_mapping.items():
            if var_col in var_data.columns:
                var_forecast[std_col] = var_data[var_col]
        if not isinstance(var_forecast.index, pd.DatetimeIndex):
            var_forecast.index = pd.date_range(start='2014-01-01', periods=len(var_forecast), freq='Q')
        return var_forecast
    except Exception as e:
        st.warning(f"Error loading VAR forecast: {str(e)}")
        return None

def load_treasury_forecast(historical_data=None):
    """Load Treasury average forecasts for the UK from the HMT Excel file, using Feb 2014 forecast for 2014 and 2015 from 'GDP' and 'CPI' sheets. Apply same transformations as VAR: CPI to index, GDP growth to level."""
    try:
        file_path = 'data/raw_data/HMT/Database_of_Average_Forecasts_for_the_UK_Economy.xlsx'
        # GDP sheet
        gdp_df = pd.read_excel(file_path, sheet_name='GDP')
        cpi_df = pd.read_excel(file_path, sheet_name='CPI')
        # Find the row for February 2014 (or closest to it)
        def get_feb_2014_row(df):
            date_col = df.columns[0]  # first column is date
            df['_parsed_date'] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True)
            target = pd.Timestamp('2014-02-01')
            idx = (df['_parsed_date'] - target).abs().idxmin()
            return df.loc[idx]
        gdp_row = get_feb_2014_row(gdp_df)
        cpi_row = get_feb_2014_row(cpi_df)
        # Find columns for 2014 and 2015 forecasts
        def get_gdp_forecast_val(row, year):
            for col in row.index:
                if str(col).strip() == str(year):
                    try:
                        val = float(row[col])
                        if not pd.isna(val):
                            return val
                    except Exception:
                        continue
            return None
        def get_cpi_forecast_val(row, year):
            for col in row.index:
                if str(year) in str(col):
                    try:
                        val = float(row[col])
                        if not pd.isna(val):
                            return val
                    except Exception:
                        continue
            return None
        gdp_2014 = get_gdp_forecast_val(gdp_row, 2014)
        gdp_2015 = get_gdp_forecast_val(gdp_row, 2015)
        cpi_2014 = get_cpi_forecast_val(cpi_row, 2014)
        cpi_2015 = get_cpi_forecast_val(cpi_row, 2015)
        # Build DataFrame: index as end of Q1 for each year
        idx = [pd.Timestamp('2014-03-31'), pd.Timestamp('2015-03-31')]
        treasury = pd.DataFrame({
            'Gross Output': [gdp_2014, gdp_2015],
            'CPI': [cpi_2014, cpi_2015]
        }, index=idx)
        # --- Apply same transformations as VAR forecast ---
        # CPI: turn into index starting at 100
        if 'CPI' in treasury.columns:
            cpi_index = [100]
            for infl in treasury['CPI'].iloc[1:]:
                cpi_index.append(cpi_index[-1] * (1 + infl/100))
            treasury['CPI'] = cpi_index[:len(treasury)]
        # Gross Output: use 2013Q4 historical value as base, apply GDP growth
        if 'Gross Output' in treasury.columns and historical_data is not None:
            base_date = pd.Timestamp('2013-10-01')
            if base_date in historical_data.index:
                base_val = historical_data.loc[base_date, 'Gross Output']
            else:
                base_val = historical_data['Gross Output'].iloc[-1]  # fallback
            go_index = [base_val]
            for growth in treasury['Gross Output'].iloc[1:]:
                go_index.append(go_index[-1] * (1 + growth/100))
            treasury['Gross Output'] = go_index[:len(treasury)]
        return treasury
    except Exception as e:
        st.warning(f"Error loading Treasury forecast: {str(e)}")
        return None

def calculate_statistics(runs_df):
    """Calculate statistics for each variable, including 75% and 25% quantiles"""
    stats = {}
    for var_key, var_name in VARIABLES_OF_INTEREST.items():
        var_stats = runs_df[var_name].groupby(level='time').agg([
            ('mean', 'mean'),
            ('std', 'std'),
            ('q10', lambda x: x.quantile(0.1)),
            ('q25', lambda x: x.quantile(0.25)),
            ('q75', lambda x: x.quantile(0.75)),
            ('q90', lambda x: x.quantile(0.9))
        ])
        stats[var_key] = var_stats
    return stats

def find_closest_run(runs_df, criterion, var_name='Gross Output'):
    """Find the run closest to the specified criterion for the selected variable"""
    if criterion == "Random Run":
        # Get all unique run indices and select one randomly
        run_indices = runs_df.index.get_level_values('run').unique()
        return np.random.choice(run_indices)
    elif criterion == "Closest to Mean":
        mean_series = runs_df[var_name].groupby(level='time').mean()
        distances = []
        for run in runs_df.index.get_level_values('run').unique():
            run_series = runs_df.loc[run, var_name]
            distance = np.mean((run_series - mean_series) ** 2)
            distances.append((run, distance))
        return min(distances, key=lambda x: x[1])[0]
    elif criterion == "Closest to Median":
        median_series = runs_df[var_name].groupby(level='time').median()
        distances = []
        for run in runs_df.index.get_level_values('run').unique():
            run_series = runs_df.loc[run, var_name]
            distance = np.mean((run_series - median_series) ** 2)
            distances.append((run, distance))
        return min(distances, key=lambda x: x[1])[0]
    elif criterion == "90% CI":
        q90 = runs_df[var_name].groupby(level='time').quantile(0.9)
        distances = []
        for run in runs_df.index.get_level_values('run').unique():
            run_series = runs_df.loc[run, var_name]
            distance = np.mean((run_series - q90) ** 2)
            distances.append((run, distance))
        return min(distances, key=lambda x: x[1])[0]
    elif criterion == "75% CI":
        q75 = runs_df[var_name].groupby(level='time').quantile(0.75)
        distances = []
        for run in runs_df.index.get_level_values('run').unique():
            run_series = runs_df.loc[run, var_name]
            distance = np.mean((run_series - q75) ** 2)
            distances.append((run, distance))
        return min(distances, key=lambda x: x[1])[0]
    elif criterion == "25% CI":
        q25 = runs_df[var_name].groupby(level='time').quantile(0.25)
        distances = []
        for run in runs_df.index.get_level_values('run').unique():
            run_series = runs_df.loc[run, var_name]
            distance = np.mean((run_series - q25) ** 2)
            distances.append((run, distance))
        return min(distances, key=lambda x: x[1])[0]
    elif criterion == "10% CI":
        q10 = runs_df[var_name].groupby(level='time').quantile(0.1)
        distances = []
        for run in runs_df.index.get_level_values('run').unique():
            run_series = runs_df.loc[run, var_name]
            distance = np.mean((run_series - q10) ** 2)
            distances.append((run, distance))
        return min(distances, key=lambda x: x[1])[0]

def get_summary_path(runs_df, path_type, var_name='Gross Output'):
    """Get the actual mean or median path across all runs for all variables"""
    summary_paths = {}
    for var in VARIABLES_OF_INTEREST.values():
        if path_type == "Mean Path":
            summary_paths[var] = runs_df[var].groupby(level='time').mean()
        elif path_type == "Median Path":
            summary_paths[var] = runs_df[var].groupby(level='time').median()
    return pd.DataFrame(summary_paths)

def create_plot(historical_data, mc_stats, selected_run, var_key, var_name, forecast_start, show_var=False, var_forecast=None, alt_label=None):
    """Create a single plot for a variable"""
    fig = go.Figure()
    # Align forecast to simulation start (2014-Q1)
    sim_start = pd.Timestamp('2014-01-01')
    forecast_dates = pd.date_range(
        start=sim_start,
        periods=len(mc_stats[var_key]),
        freq='QE'
    )
    # Add background color for forecast period
    fig.add_vrect(
        x0=forecast_dates[0], x1=forecast_dates[-1],
        fillcolor="rgba(200,200,255,0.2)", layer="below", line_width=0
    )
    # Subtle annotation for input data (historical period)
    fig.add_annotation(
        x=historical_data.index[0],
        y=1.01, xref="x", yref="paper",
        text="Input Data",
        showarrow=False,
        font=dict(size=11, color="gray"),
        align="left", bgcolor="rgba(240,240,240,0.5)", bordercolor="rgba(200,200,200,0.5)",
        xanchor="left"
    )
    # Subtle annotation for simulation period (forecast)
    fig.add_annotation(
        x=forecast_dates[0],
        y=1.01, xref="x", yref="paper",
        text="Simulation Period",
        showarrow=False,
        font=dict(size=11, color="gray"),
        align="left", bgcolor="rgba(200,200,255,0.2)", bordercolor="rgba(200,200,200,0.2)",
        xanchor="left"
    )
    # Add less faint vertical white lines at Q1 of each year
    all_dates = pd.concat([pd.Series(historical_data.index), pd.Series(forecast_dates)]).drop_duplicates().sort_values()
    q1_dates = [d for d in all_dates if d.month == 1]
    for q1 in q1_dates:
        fig.add_vline(x=q1, line=dict(color='rgba(255,255,255,0.5)', width=1), layer='below')
    # Add 90% confidence intervals (shaded area)
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=mc_stats[var_key]['q90'],
        fill=None,
        mode='lines',
        line_color='rgba(0,100,80,0.2)',
        name='90% CI Upper'
    ))
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=mc_stats[var_key]['q10'],
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,100,80,0.2)',
        name='90% CI Lower'
    ))
    # Add 75% confidence intervals (shaded area, more transparent)
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=mc_stats[var_key]['q75'],
        fill=None,
        mode='lines',
        line_color='rgba(0,100,200,0.1)',
        name='75% CI Upper'
    ))
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=mc_stats[var_key]['q25'],
        fill='tonexty',
        mode='lines',
        line_color='rgba(255,255,255,0.1)',
        name='75% CI Lower'
    ))
    # Add selected run (forecast only)
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=selected_run[var_name].values,
        name='Selected Run',
        line=dict(color='red')
    ))
    # Add alternative forecast if requested and available
    if show_var and var_forecast is not None and var_name in var_forecast.columns:
        var_dates = var_forecast.index
        label = f"{alt_label} Forecast" if alt_label else "Alternative Forecast"
        fig.add_trace(go.Scatter(
            x=var_dates,
            y=var_forecast[var_name],
            name=label,
            line=dict(color='green', dash='dash', width=2)
        ))
    # Add historical data LAST so it is on top
    fig.add_trace(go.Scatter(
        x=historical_data.index,
        y=historical_data[var_name],
        name='Historical',
        line=dict(color='black')
    ))
    # Update layout
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title=var_key,
        showlegend=True,
        hovermode='x unified',
        xaxis=dict(tickformat='%Y-Q%q')
    )
    return fig

def compute_rmse_table(historical_data, mc_stats, selected_run, var_name, sim_start, var_forecast=None, show_var=False, alt_label=None, var_forecast_var=None, var_forecast_treasury=None):
    """Compute percentage RMSE between historical and forecast for various horizons, always showing all four columns."""
    horizons = [1, 4, 8, 12, 20]  # quarters: 1Q, 1Y, 2Y, 3Y, 5Y
    results = []
    # Find the index of the simulation start in the historical data
    if sim_start in historical_data.index:
        start_idx = historical_data.index.get_loc(sim_start)
    else:
        return pd.DataFrame()  # No matching start
    # Map variable names between historical and simulation data
    var_mapping = {
        'Gross Output': 'Gross Output',
        'CPI': 'CPI',
        'Unemployment Rate': 'Unemployment Rate',
        'Unemployment Rate (Value)': 'Unemployment Rate'
    }
    hist_var = var_name
    sim_var = var_mapping.get(var_name, var_name)
    for h in horizons:
        # Historical value at t+h
        if start_idx + h >= len(historical_data):
            hist_val = np.nan
        else:
            hist_val = historical_data[hist_var].iloc[start_idx + h]
        # Forecast mean at t+h
        if h-1 < len(mc_stats[sim_var]):
            forecast_mean = mc_stats[sim_var]['mean'].iloc[h-1]
            selected_val = selected_run[sim_var].iloc[h-1]
        else:
            forecast_mean = np.nan
            selected_val = np.nan
        # VAR forecast at t+h
        var_val = np.nan
        if var_forecast_var is not None and sim_var in var_forecast_var.columns:
            try:
                var_idx = h-1
                if var_idx < len(var_forecast_var):
                    var_val = var_forecast_var[sim_var].iloc[var_idx]
            except Exception:
                var_val = np.nan
        # Treasury forecast at t+h
        treasury_val = np.nan
        if var_forecast_treasury is not None and sim_var in var_forecast_treasury.columns:
            try:
                var_idx = h-1
                if var_idx < len(var_forecast_treasury):
                    treasury_val = var_forecast_treasury[sim_var].iloc[var_idx]
            except Exception:
                treasury_val = np.nan
        # Calculate percentage RMSE
        if not (np.isnan(forecast_mean) or np.isnan(hist_val)):
            pct_rmse_mean = math.sqrt(((forecast_mean - hist_val) / abs(hist_val)) ** 2) * 100
            pct_rmse_sel = math.sqrt(((selected_val - hist_val) / abs(hist_val)) ** 2) * 100
        else:
            pct_rmse_mean = pct_rmse_sel = np.nan
        if not (np.isnan(var_val) or np.isnan(hist_val)):
            pct_rmse_var = math.sqrt(((var_val - hist_val) / abs(hist_val)) ** 2) * 100
        else:
            pct_rmse_var = None
        if not (np.isnan(treasury_val) or np.isnan(hist_val)):
            pct_rmse_treasury = math.sqrt(((treasury_val - hist_val) / abs(hist_val)) ** 2) * 100
        else:
            pct_rmse_treasury = None
        row = {
            'Horizon': f'{h}Q',
            'Our model - mean path': pct_rmse_mean,
            'Our model - your selected path': pct_rmse_sel,
            'VAR forecast': pct_rmse_var,
            'Treasury average': pct_rmse_treasury
        }
        results.append(row)
    # Create DataFrame and format the results
    df = pd.DataFrame(results)
    # Format RMSE columns to 2 decimal places with percentage sign, and bold the best (lowest) value in each row
    rmse_cols = [col for col in df.columns if col != 'Horizon']
    def bold_min(row):
        vals = []
        for col in rmse_cols:
            try:
                val = float(row[col])
            except:
                val = float('inf')
            vals.append(val)
        min_val = min([v for v in vals if not math.isnan(v)], default=None)
        new_row = {}
        for col, val in zip(rmse_cols, vals):
            if val is None or math.isnan(val):
                new_row[col] = 'N/A'
            elif val == min_val:
                new_row[col] = f"<b>{val:.2f}%</b>"
            else:
                new_row[col] = f"{val:.2f}%"
        return pd.Series(new_row)
    if not df.empty:
        df[rmse_cols] = df.apply(bold_min, axis=1)
    return df

def main():
    st.markdown("""
        <style>
        .block-container { padding-top: 1rem !important; }
        .css-18e3th9 { padding-top: 1rem !important; }
        .css-1d391kg { padding-top: 1rem !important; }
        </style>
    """, unsafe_allow_html=True)
    st.title("Macromodel Validation")
    with st.sidebar:
        st.header("Path Selection")
        path_type = st.selectbox(
            "Select path type:",
            ["Mean Path", "Median Path", "Single Run"]
        )
        
        if path_type == "Single Run":
            path_variable = st.selectbox(
                "Variable for path selection:",
                ["Gross Output", "CPI", "Unemployment Rate"]
            )
            path_criterion = st.selectbox(
                "Select path based on:",
                ["Random Run", "Closest to Mean", "Closest to Median", "10% CI", "25% CI", "75% CI", "90% CI"]
            )
        else:
            path_variable = "Gross Output"  # Default variable for mean/median paths
            path_criterion = path_type  # Use the path type as the criterion

        st.markdown("""
            <div style='height: 2rem;'></div>
        """, unsafe_allow_html=True)
        alt_forecast = st.selectbox(
            "Show alternative forecasts?",
            ["None", "VAR", "Treasury Average"]
        )
        st.markdown("""
            <div style='height: 2rem;'></div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        results_dir = st.text_input(
            "Results Directory",
            value="output-MonteCarlo-GBR/monte_carlo"
        )
    try:
        full_hist_data = load_historical_data.__wrapped__()
        sim_start = pd.Timestamp('2014-01-01')
        if sim_start in full_hist_data.index:
            start_idx = full_hist_data.index.get_loc(sim_start) - 20
            if start_idx < 0:
                start_idx = 0
            hist_window = full_hist_data.iloc[start_idx:]
        else:
            hist_window = full_hist_data
        mc_results = load_monte_carlo_results(results_dir)
        mc_stats = calculate_statistics(mc_results)
        # Always load both VAR and Treasury forecasts for the RMSE table
        var_forecast_df = load_var_forecast(results_dir, historical_data=full_hist_data)
        treasury_forecast_df = load_treasury_forecast(historical_data=full_hist_data)
        # Only plot the selected alternative forecast
        alt_forecast_df = None
        alt_label = None
        if alt_forecast == "VAR":
            alt_forecast_df = var_forecast_df
            alt_label = "VAR"
        elif alt_forecast == "Treasury Average":
            alt_forecast_df = treasury_forecast_df
            alt_label = "Treasury"
        
        # Handle path selection
        if path_type in ["Mean Path", "Median Path"]:
            # Get the actual mean/median path for all variables
            selected_run = get_summary_path(mc_results, path_type, path_variable)
        else:
            # Find the closest run to the selected criterion
            selected_run_idx = find_closest_run(mc_results, path_criterion, var_name=path_variable)
            selected_run = mc_results.loc[selected_run_idx]
        
        tab_labels = list(VARIABLES_OF_INTEREST.keys())
        tabs = st.tabs(tab_labels)
        for i, (var_key, var_name) in enumerate(VARIABLES_OF_INTEREST.items()):
            with tabs[i]:
                fig = create_plot(
                    hist_window,
                    mc_stats,
                    selected_run,
                    var_key,
                    var_name,
                    forecast_start=hist_window.index[-1],
                    show_var=(alt_forecast != "None"),
                    var_forecast=alt_forecast_df,
                    alt_label=alt_label
                )
                st.plotly_chart(fig, use_container_width=True)
                rmse_table = compute_rmse_table(
                    hist_window, mc_stats, selected_run, var_name, sim_start,
                    var_forecast=alt_forecast_df, show_var=(alt_forecast != "None"), alt_label=alt_label,
                    var_forecast_var=var_forecast_df, var_forecast_treasury=treasury_forecast_df
                )
                st.markdown('**Forecast RMSE vs. alternative forecasts and what actually happened**')
                st.write(rmse_table.to_html(escape=False, index=False), unsafe_allow_html=True)
                st.info("""RMSE (Root Mean Squared Error) is a standard measure of model fit. It quantifies the average magnitude of the error between model forecasts and actual observed values, with lower values indicating better fit. All values are shown as a percentage of the actual value.""")
                st.info("""The VAR forecast uses historic data from the years prior to the simulation run to forecast the same period as the INET Oxford macromodel. Our VAR implementation follows Poledna et al 2023.""")
                st.info("""The Treasury average forecast is the mean of a range of forecasters made by private sector forecasters in the first period of our simulation window. See the official [Treasury forecast database](https://www.gov.uk/government/statistics/database-of-forecasts-for-the-uk-economy) for more information.""")

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

if __name__ == "__main__":
    main() 