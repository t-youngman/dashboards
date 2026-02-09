import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from macro_data import DataWrapper

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
        df.index = pd.date_range(start=f'{start_year}-01-01', periods=len(df), freq='QE')
    # Rebase CPI so that the first simulation period (2014-Q1) is 100
    sim_start = pd.Timestamp('2014-01-01')
    if 'CPI' in df.columns and sim_start in df.index:
        cpi_base = df.loc[sim_start, 'CPI']
        df['CPI'] = df['CPI'] / cpi_base * 100
    return df

@st.cache_data
def load_monte_carlo_results(results_dir):
    """Load and process Monte Carlo simulation results, and use nominal Gross Output levels.
    Discovers both UK_run_*.csv (Monte Carlo) and GBR_shallow_output_*.csv (e.g. output_baseline)."""
    results_dir = Path(results_dir).resolve()
    files = sorted(
        list(results_dir.glob("UK_run_*.csv")) + list(results_dir.glob("GBR_shallow_output_*.csv"))
    )
    all_runs = []
    for file in files:
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
                var_forecast.index = pd.date_range(start=start, periods=len(var_forecast), freq='QE')
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
            var_forecast.index = pd.date_range(start='2014-01-01', periods=len(var_forecast), freq='QE')
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

def create_plot(historical_data, mc_stats, selected_run, var_key, var_name, forecast_start, show_var=False, var_forecast=None, alt_label=None,
               compare_stats=None, compare_run=None, primary_label="Baseline", compare_label="Shock", shock_start_q=0, show_shock_annotation=False):
    """Create a single plot for a variable. If compare_stats/compare_run are set, add second set with distinct colors."""
    fig = go.Figure()
    sim_start = pd.Timestamp('2014-01-01')
    forecast_dates = pd.date_range(
        start=sim_start,
        periods=len(mc_stats[var_key]),
        freq='QE'
    )
    fig.add_vrect(
        x0=forecast_dates[0], x1=forecast_dates[-1],
        fillcolor="rgba(200,255,200,0.2)", layer="below", line_width=0
    )
    if show_shock_annotation:
        shock_idx = max(0, min(shock_start_q, len(forecast_dates) - 1))
        fig.add_vrect(
            x0=forecast_dates[shock_idx], x1=forecast_dates[-1],
            fillcolor="rgba(255,210,100,0.2)", layer="below", line_width=0
        )
        fig.add_annotation(
            x=forecast_dates[shock_idx],
            y=1.01, xref="x", yref="paper",
            text="Shock",
            showarrow=False,
            font=dict(size=11, color="gray"),
            align="left", bgcolor="rgba(200,255,200,0.2)", bordercolor="rgba(200,200,200,0.2)",
            xanchor="left"
        )
    fig.add_annotation(
        x=historical_data.index[0],
        y=1.01, xref="x", yref="paper",
        text="Input Data",
        showarrow=False,
        font=dict(size=11, color="gray"),
        align="left", bgcolor="rgba(240,240,240,0.5)", bordercolor="rgba(200,200,200,0.5)",
        xanchor="left"
    )
    fig.add_annotation(
        x=forecast_dates[0],
        y=1.01, xref="x", yref="paper",
        text="Simulation",
        showarrow=False,
        font=dict(size=11, color="gray"),
        align="left", bgcolor="rgba(200,200,255,0.2)", bordercolor="rgba(200,200,200,0.2)",
        xanchor="left"
    )

    all_dates = pd.concat([pd.Series(historical_data.index), pd.Series(forecast_dates)]).drop_duplicates().sort_values()
    q1_dates = [d for d in all_dates if d.month == 1]
    for q1 in q1_dates:
        fig.add_vline(x=q1, line=dict(color='rgba(255,255,255,0.5)', width=1), layer='below')

    # Primary set: 90% and 75% CI, then selected run
    p = primary_label if (compare_stats is not None) else ""
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=mc_stats[var_key]['q90'], fill=None, mode='lines',
        line_color='rgba(0,100,80,0.2)', name=f'{p} 90% CI Upper'.strip() or '90% CI Upper'
    ))
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=mc_stats[var_key]['q10'], fill='tonexty', mode='lines',
        line_color='rgba(0,100,80,0.2)', name=f'{p} 90% CI Lower'.strip() or '90% CI Lower'
    ))
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=mc_stats[var_key]['q75'], fill=None, mode='lines',
        line_color='rgba(0,100,200,0.1)', name=f'{p} 75% CI Upper'.strip() or '75% CI Upper'
    ))
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=mc_stats[var_key]['q25'], fill='tonexty', mode='lines',
        line_color='rgba(255,255,255,0.1)', name=f'{p} 75% CI Lower'.strip() or '75% CI Lower'
    ))
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=selected_run[var_name].values,
        name=f'{p} Selected Run'.strip() or 'Selected Run',
        line=dict(color='green')
    ))

    # Comparison set (optional)
    if compare_stats is not None and compare_run is not None and var_key in compare_stats:
        n_compare = len(compare_stats[var_key])
        compare_dates = pd.date_range(start=sim_start, periods=n_compare, freq='QE')
        fig.add_trace(go.Scatter(
            x=compare_dates, y=compare_stats[var_key]['q90'], fill=None, mode='lines',
            line_color='rgba(180,80,0,0.35)', name=f'{compare_label} 90% CI Upper'
        ))
        fig.add_trace(go.Scatter(
            x=compare_dates, y=compare_stats[var_key]['q10'], fill='tonexty', mode='lines',
            line_color='rgba(180,80,0,0.35)', name=f'{compare_label} 90% CI Lower'
        ))
        fig.add_trace(go.Scatter(
            x=compare_dates, y=compare_stats[var_key]['q75'], fill=None, mode='lines',
            line_color='rgba(220,140,0,0.15)', name=f'{compare_label} 75% CI Upper'
        ))
        fig.add_trace(go.Scatter(
            x=compare_dates, y=compare_stats[var_key]['q25'], fill='tonexty', fillcolor='rgba(255,255,255,0.2)', mode='lines',
            line_color='rgba(220,140,0,0.15)', name=f'{compare_label} 75% CI Lower'
        ))
        compare_vals = compare_run[var_name].values[:n_compare] if len(compare_run[var_name]) >= n_compare else compare_run[var_name].values
        fig.add_trace(go.Scatter(
            x=compare_dates[:len(compare_vals)], y=compare_vals,
            name=f'{compare_label} Selected Run',
            line=dict(color='red', dash='dash', width=2)
        ))

    if show_var and var_forecast is not None and var_name in var_forecast.columns:
        var_dates = var_forecast.index
        label = f"{alt_label} Forecast" if alt_label else "Alternative Forecast"
        fig.add_trace(go.Scatter(
            x=var_dates, y=var_forecast[var_name], name=label,
            line=dict(color='green', dash='dash', width=2)
        ))
    fig.add_trace(go.Scatter(
        x=historical_data.index, y=historical_data[var_name],
        name='Historical', line=dict(color='black')
    ))
    fig.update_layout(
        xaxis_title="Time", yaxis_title=var_key,
        showlegend=True, hovermode='x unified',
        xaxis=dict(tickformat='%Y-Q%q')
    )
    return fig

def compute_rmse_table(historical_data, mc_stats, selected_run, var_name, sim_start, var_forecast=None, show_var=False, alt_label=None, var_forecast_var=None, var_forecast_treasury=None,
                       compare_stats=None, compare_run=None, primary_label="Baseline", compare_label="Shock"):
    """Compute percentage RMSE for various horizons. If compare_* are set, include comparison set columns."""
    horizons = [1, 4, 8, 12, 20]
    results = []
    if sim_start in historical_data.index:
        start_idx = historical_data.index.get_loc(sim_start)
    else:
        return pd.DataFrame()
    var_mapping = {
        'Gross Output': 'Gross Output',
        'CPI': 'CPI',
        'Unemployment Rate': 'Unemployment Rate',
        'Unemployment Rate (Value)': 'Unemployment Rate'
    }
    hist_var = var_name
    sim_var = var_mapping.get(var_name, var_name)
    for h in horizons:
        if start_idx + h >= len(historical_data):
            hist_val = np.nan
        else:
            hist_val = historical_data[hist_var].iloc[start_idx + h]
        if h-1 < len(mc_stats[sim_var]):
            forecast_mean = mc_stats[sim_var]['mean'].iloc[h-1]
            selected_val = selected_run[sim_var].iloc[h-1]
        else:
            forecast_mean = selected_val = np.nan
        var_val = np.nan
        if var_forecast_var is not None and sim_var in var_forecast_var.columns:
            try:
                if h-1 < len(var_forecast_var):
                    var_val = var_forecast_var[sim_var].iloc[h-1]
            except Exception:
                var_val = np.nan
        treasury_val = np.nan
        if var_forecast_treasury is not None and sim_var in var_forecast_treasury.columns:
            try:
                if h-1 < len(var_forecast_treasury):
                    treasury_val = var_forecast_treasury[sim_var].iloc[h-1]
            except Exception:
                treasury_val = np.nan
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
            f'{primary_label} - mean' if compare_stats else 'Our model - mean path': pct_rmse_mean,
            f'{primary_label} - selected' if compare_stats else 'Our model - your selected path': pct_rmse_sel,
            'VAR forecast': pct_rmse_var,
            'Treasury average': pct_rmse_treasury
        }
        if compare_stats is not None and compare_run is not None and sim_var in compare_stats:
            if h-1 < len(compare_stats[sim_var]):
                comp_mean = compare_stats[sim_var]['mean'].iloc[h-1]
                comp_sel = compare_run[sim_var].iloc[h-1]
            else:
                comp_mean = comp_sel = np.nan
            if not (np.isnan(comp_mean) or np.isnan(hist_val)):
                pct_rmse_comp_mean = math.sqrt(((comp_mean - hist_val) / abs(hist_val)) ** 2) * 100
            else:
                pct_rmse_comp_mean = np.nan
            if not (np.isnan(comp_sel) or np.isnan(hist_val)):
                pct_rmse_comp_sel = math.sqrt(((comp_sel - hist_val) / abs(hist_val)) ** 2) * 100
            else:
                pct_rmse_comp_sel = np.nan
            row[f'{compare_label} - mean'] = pct_rmse_comp_mean
            row[f'{compare_label} - selected'] = pct_rmse_comp_sel
        results.append(row)
    df = pd.DataFrame(results)
    rmse_cols = [col for col in df.columns if col != 'Horizon']
    def bold_min(row):
        vals = []
        for col in rmse_cols:
            try:
                val = float(row[col])
            except (TypeError, ValueError):
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

def build_selected_run_pct_change_table(run_a, run_b, var_name, label_b_vs_a):
    """Build a table of period-by-period percentage change (run_b vs run_a) for one variable."""
    n = min(len(run_a), len(run_b))
    if n == 0 or var_name not in run_a.columns or var_name not in run_b.columns:
        return pd.DataFrame()
    dates = pd.date_range(start=pd.Timestamp('2014-01-01'), periods=n, freq='QE')
    rows = []
    for i in range(n):
        a_val = run_a[var_name].iloc[i]
        b_val = run_b[var_name].iloc[i]
        if a_val != 0 and not np.isnan(a_val):
            pct = (b_val - a_val) / abs(a_val) * 100
        else:
            pct = np.nan
        q = (dates[i].month - 1) // 3 + 1
        rows.append({'Period': f"{dates[i].year}-Q{q}", label_b_vs_a: pct})
    return pd.DataFrame(rows)

def create_single_var_comparison_chart(run_a, run_b, label_a, label_b, var_name):
    """Create a single chart comparing the two selected runs for one variable."""
    n = min(len(run_a), len(run_b))
    if n == 0 or var_name not in run_a.columns or var_name not in run_b.columns:
        return go.Figure()
    dates = pd.date_range(start=pd.Timestamp('2014-01-01'), periods=n, freq='QE')
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=run_a[var_name].values[:n], name=label_a, line=dict(color='red')
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=run_b[var_name].values[:n], name=label_b, line=dict(color='blue', dash='dash')
    ))
    fig.update_layout(yaxis_title=var_name, showlegend=True, hovermode='x unified')
    return fig

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
            ["Median Path", "Mean Path", "Single Run"]
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
        compare_mode = st.checkbox("Compare with second run set", value=False)
        results_dir = st.text_input(
            "Results Directory",
            value="output-MonteCarlo-GBR/monte_carlo",
            help="Primary set of runs (UK_run_*.csv or GBR_shallow_output_*.csv)."
        )
        primary_label = ""
        compare_dir = None
        compare_label = "Shock"
        if compare_mode:
            primary_label = st.text_input("Label for primary set", value="Baseline")
            compare_dir = st.text_input(
                "Comparison results directory",
                value="output_baseline",
                help="Second set of runs to compare."
            )
            compare_label = st.text_input("Label for comparison set", value="Shock")
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
        n_periods_total = len(mc_results.index.get_level_values('time').unique())
        max_periods = st.sidebar.number_input(
            "Max time periods to display",
            min_value=1,
            max_value=n_periods_total,
            value=n_periods_total,
            help="Limit simulation periods shown in charts and tables (default: all)."
        )
        show_shock_annotation = st.sidebar.checkbox("Add shock annotation?", value=False)
        shock_start_q = 0
        if show_shock_annotation:
            shock_start_q = st.sidebar.number_input(
                "Shock start quarter",
                min_value=0,
                max_value=n_periods_total - 1 if n_periods_total else 0,
                value=0,
                help="Quarter index (0-based) when shock starts; 0 = first simulation period."
            )
        mc_stats = calculate_statistics(mc_results)
        var_forecast_df = load_var_forecast(results_dir, historical_data=full_hist_data)
        treasury_forecast_df = load_treasury_forecast(historical_data=full_hist_data)
        alt_forecast_df = None
        alt_label = None
        if alt_forecast == "VAR":
            alt_forecast_df = var_forecast_df
            alt_label = "VAR"
        elif alt_forecast == "Treasury Average":
            alt_forecast_df = treasury_forecast_df
            alt_label = "Treasury"

        if path_type in ["Mean Path", "Median Path"]:
            selected_run = get_summary_path(mc_results, path_type, path_variable)
        else:
            selected_run_idx = find_closest_run(mc_results, path_criterion, var_name=path_variable)
            selected_run = mc_results.loc[selected_run_idx]

        mc_stats_2 = None
        selected_run_2 = None
        if compare_mode and compare_dir:
            mc_results_2 = load_monte_carlo_results(compare_dir)
            mc_stats_2 = calculate_statistics(mc_results_2)
            if path_type in ["Mean Path", "Median Path"]:
                selected_run_2 = get_summary_path(mc_results_2, path_type, path_variable)
            else:
                sel_idx_2 = find_closest_run(mc_results_2, path_criterion, var_name=path_variable)
                selected_run_2 = mc_results_2.loc[sel_idx_2]

        # Apply max_periods limit to simulation data
        mc_stats = {k: v.iloc[:max_periods] for k, v in mc_stats.items()}
        selected_run = selected_run.iloc[:max_periods]
        if mc_stats_2 is not None:
            mc_stats_2 = {k: v.iloc[:max_periods] for k, v in mc_stats_2.items()}
        if selected_run_2 is not None:
            selected_run_2 = selected_run_2.iloc[:max_periods]
        if alt_forecast_df is not None and len(alt_forecast_df) > max_periods:
            alt_forecast_df = alt_forecast_df.iloc[:max_periods]
        if var_forecast_df is not None and len(var_forecast_df) > max_periods:
            var_forecast_df = var_forecast_df.iloc[:max_periods]
        if treasury_forecast_df is not None and len(treasury_forecast_df) > max_periods:
            treasury_forecast_df = treasury_forecast_df.iloc[:max_periods]

        tab_labels = list(VARIABLES_OF_INTEREST.keys())
        tabs = st.tabs(tab_labels)
        for i, (var_key, var_name) in enumerate(VARIABLES_OF_INTEREST.items()):
            with tabs[i]:
                fig = create_plot(
                    hist_window, mc_stats, selected_run, var_key, var_name,
                    forecast_start=hist_window.index[-1],
                    show_var=(alt_forecast != "None"), var_forecast=alt_forecast_df, alt_label=alt_label,
                    compare_stats=mc_stats_2, compare_run=selected_run_2,
                    primary_label=primary_label, compare_label=compare_label,
                    shock_start_q=shock_start_q, show_shock_annotation=show_shock_annotation
                )
                st.plotly_chart(fig, width='stretch')
                rmse_table = compute_rmse_table(
                    hist_window, mc_stats, selected_run, var_name, sim_start,
                    var_forecast=alt_forecast_df, show_var=(alt_forecast != "None"), alt_label=alt_label,
                    var_forecast_var=var_forecast_df, var_forecast_treasury=treasury_forecast_df,
                    compare_stats=mc_stats_2, compare_run=selected_run_2,
                    primary_label=primary_label, compare_label=compare_label
                )
                st.markdown('**Forecast RMSE vs. alternative forecasts and what actually happened**')
                st.write(rmse_table.to_html(escape=False, index=False), unsafe_allow_html=True)
                st.info("""RMSE (Root Mean Squared Error) is a standard measure of model fit. It quantifies the average magnitude of the error between model forecasts and actual observed values, with lower values indicating better fit. All values are shown as a percentage of the actual value.""")
                st.info("""The VAR forecast uses historic data from the years prior to the simulation run to forecast the same period as the INET Oxford macromodel. Our VAR implementation follows Poledna et al 2023.""")
                st.info("""The Treasury average forecast is the mean of a range of forecasters made by private sector forecasters in the first period of our simulation window. See the official [Treasury forecast database](https://www.gov.uk/government/statistics/database-of-forecasts-for-the-uk-economy) for more information.""")

                if compare_mode and selected_run_2 is not None:
                    st.markdown("---")
                    st.subheader("Selected run comparison")
                    pct_label = f"% change ({compare_label} vs {primary_label})"
                    comp_table = build_selected_run_pct_change_table(
                        selected_run, selected_run_2, var_name, pct_label
                    )
                    if not comp_table.empty:
                        styled = comp_table.style.format({pct_label: "{:+.2f}%"}, na_rep="—")
                        st.dataframe(styled, width='stretch')
                        fig_comp = create_single_var_comparison_chart(
                            selected_run, selected_run_2, primary_label, compare_label, var_name
                        )
                        st.plotly_chart(fig_comp, width='stretch')

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

if __name__ == "__main__":
    main() 