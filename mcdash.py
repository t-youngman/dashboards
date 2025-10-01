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
    'Gross Domestic Product': 'Gross Domestic Product',
    'CPI': 'CPI',
    'Unemployment Rate': 'Unemployment Rate'
}
CONFIDENCE_LEVELS = {
    '90%': 0.90,
    '10%': 0.10
}

def construct_gdp_estimate(exo, country="GBR"):
    """Construct historical GDP estimate using available national accounts variables.
    
    This function attempts to construct GDP using multiple approaches based on the
    available variables in the exogenous national accounts data. It follows the
    same GDP calculation approaches used in the macromodel.
    
    Args:
        exo: Exogenous data object containing national_accounts
        country: Country code for error reporting
        
    Returns:
        pd.Series: GDP estimate time series
    """
    
    # Try different approaches to construct GDP
    gdp_estimates = {}
    
    # Approach 2: GDP components approach (expenditure method)
    gdp_components = []
    component_names = []
    
    # Look for common GDP components in national accounts
    potential_components = [
        'Household Consumption (Value)',
        'Government Consumption (Value)', 
        'Real Government Consumption (Value)',
        'Gross Fixed Capital Formation (Value)',
        'Changes in Inventories (Value)',
        'Exports (Value)',
        'Imports (Value)',
        'Net Exports (Value)',
        'Final Consumption Expenditure (Value)',
        'Gross Capital Formation (Value)',
        'Real Household Consumption (Value)',
        'Real Household Investment (Value)'
    ]
    
    for component in potential_components:
        if component in exo.national_accounts:
            gdp_components.append(exo.national_accounts[component])
            component_names.append(component)
    
    # If we have enough components, try to construct GDP using expenditure approach
    if len(gdp_components) >= 3:  # Need at least 3 major components
        try:
            # Look for consumption, investment, government, exports, imports
            consumption = None
            investment = None
            government = None
            exports = None
            imports = None
            
            for i, name in enumerate(component_names):
                if 'Consumption' in name and 'Household' in name:
                    consumption = gdp_components[i]
                elif 'Consumption' in name and 'Government' in name:
                    government = gdp_components[i]
                elif 'Capital Formation' in name or 'Investment' in name:
                    investment = gdp_components[i]
                elif 'Exports' in name and 'Imports' not in name:
                    exports = gdp_components[i]
                elif 'Imports' in name and 'Exports' not in name:
                    imports = gdp_components[i]
            
            # Construct GDP using expenditure approach: C + I + G + (X - M)
            if consumption is not None and investment is not None and government is not None:
                gdp_expenditure = consumption + investment + government
                if exports is not None and imports is not None:
                    gdp_expenditure += (exports - imports)
                elif exports is not None:
                    gdp_expenditure += exports
                elif imports is not None:
                    gdp_expenditure -= imports
                
                gdp_estimates['Expenditure Approach'] = gdp_expenditure
                
        except Exception as e:
            print(f"Error constructing expenditure approach GDP for {country}: {e}")
    
    # Approach 3: Try to find direct GDP measures
    direct_gdp_vars = [
        'GDP (Value)',
        'Gross Domestic Product (Value)',
        'GDP at Market Prices (Value)',
        'GDP at Current Prices (Value)',
        'Real GDP (Value)',
        'Nominal GDP (Value)'
    ]
    
    for var in direct_gdp_vars:
        if var in exo.national_accounts:
            gdp_estimates[f'Direct {var}'] = exo.national_accounts[var]
    
    # Approach 4: Try output approach (Value Added)
    # Look for value added measures
    value_added_vars = [
        'Value Added (Value)',
        'Gross Value Added (Value)'
    ]
    
    for var in value_added_vars:
        if var in exo.national_accounts:
            gdp_estimates[f'Output Approach ({var})'] = exo.national_accounts[var]
            break
    
    # Choose the best available estimate
    if gdp_estimates:
        # Prefer direct measures, then output approach, then expenditure approach
        preferred_order = [
            'Direct GDP (Value)', 
            'Direct Gross Domestic Product (Value)', 
            'Direct GDP at Market Prices (Value)', 
            'Direct GDP at Current Prices (Value)',
            'Direct Real GDP (Value)',
            'Direct Nominal GDP (Value)',
            'Output Approach (Gross Value Added (Value))',
            'Output Approach (Value Added (Value))',
            'Expenditure Approach'
        ]
        
        for preferred in preferred_order:
            if preferred in gdp_estimates:
                return gdp_estimates[preferred]
        
        # If none of the preferred ones are available, return the first one
        return list(gdp_estimates.values())[0]
    else:
        # Fallback: return a series of NaN values
        print(f"Warning: No GDP estimate could be constructed for {country}")
        if len(exo.national_accounts) > 0:
            # Use the length of the first available column
            first_col = list(exo.national_accounts.columns)[0]
            return pd.Series([np.nan] * len(exo.national_accounts[first_col]), 
                           index=exo.national_accounts.index)
        else:
            return pd.Series([np.nan])

@st.cache_data
def load_historical_data(country="GBR"):
    """Load the full exogenous data series for historical data."""
    data = DataWrapper.init_from_pickle("data/processed_data/data.pkl")
    country_data = data.synthetic_countries[country]
    exo = country_data.exogenous_data

    # Construct GDP estimate using available variables
    gdp_estimate = construct_gdp_estimate(exo, country)

    # Use the actual index from the national_accounts data
    df_index = exo.national_accounts.index
    
    # Create DataFrame with proper alignment
    df = pd.DataFrame(index=df_index)
    df['Gross Domestic Product'] = gdp_estimate
    df['CPI'] = exo.national_accounts['CPI (Value)'] if 'CPI (Value)' in exo.national_accounts else np.nan
    
    # Handle unemployment rate - it might have a different index, so we need to align it
    if hasattr(exo, 'labour_stats') and 'Unemployment Rate (Value)' in exo.labour_stats:
        # Try to align the unemployment data with the national accounts index
        unemp_data = exo.labour_stats['Unemployment Rate (Value)']
        if hasattr(unemp_data, 'reindex'):
            # If it has a reindex method, use it to align with national accounts
            df['Unemployment Rate'] = unemp_data.reindex(df_index, method='ffill')
        else:
            # Fallback: create a series with the same index
            df['Unemployment Rate'] = pd.Series([unemp_data.iloc[0]] * len(df_index), index=df_index)
    else:
        df['Unemployment Rate'] = np.nan
    
    df = df.ffill().bfill()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill()
    # Rebase CPI so that the first simulation period (2014-Q1) is 100
    sim_start = pd.Timestamp('2014-01-01')
    if 'CPI' in df.columns and sim_start in df.index:
        cpi_base = df.loc[sim_start, 'CPI']
        df['CPI'] = df['CPI'] / cpi_base * 100
    return df

@st.cache_data
def load_monte_carlo_results(results_dir):
    """Load and process Monte Carlo simulation results from shallow output files, and use nominal Gross Output levels"""
    results_dir = Path(results_dir).resolve()
    all_runs = []
    for file in sorted(results_dir.glob("UK_run_*.csv")):
        try:
            run_data = pd.read_csv(file, index_col=0)
            # Use GDP if available, otherwise skip this run
            if 'Gross Domestic Product' not in run_data.columns:
                st.warning(f"No GDP data found in {file}, skipping...")
                continue
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
def load_monte_carlo_headlines(results_dir, country="GBR", filename=None):
    """Load and process Monte Carlo simulation results from headlines files (CSV format with Simulation, Timestep, Country columns)"""
    results_dir = Path(results_dir).resolve()
    
    # If a specific filename is provided, use it
    if filename:
        headlines_file = results_dir / filename
        if not headlines_file.exists():
            raise ValueError(f"Specified file {filename} not found in {results_dir}")
    else:
        # Look for CSV files that might contain Monte Carlo headlines
        csv_files = list(results_dir.glob("*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in {results_dir}")
        
        # Try to find a file that looks like Monte Carlo headlines (has Simulation, Timestep, and Country columns)
        headlines_file = None
        for file in csv_files:
            try:
                # Read just the header to check column structure
                sample_df = pd.read_csv(file, nrows=0)
                if ('Simulation' in sample_df.columns and 'Timestep' in sample_df.columns and 
                    'Country' in sample_df.columns):
                    headlines_file = file
                    break
            except Exception:
                continue
        
        if headlines_file is None:
            raise ValueError(f"No Monte Carlo headlines file found in {results_dir}. Expected CSV with 'Simulation', 'Timestep', and 'Country' columns.")
    
    try:
        # Load the headlines data
        headlines_data = pd.read_csv(headlines_file)
        
        # Filter for the selected country's data
        country_data = headlines_data[headlines_data['Country'] == country].copy()
        if country_data.empty:
            raise ValueError(f"No {country} data found in {headlines_file}")
        
        # Check if we have the expected variable columns
        expected_vars = ['GDP', 'CPI', 'Unemployment Rate']
        available_vars = [var for var in expected_vars if var in country_data.columns]
        if not available_vars:
            raise ValueError(f"None of the expected variables {expected_vars} found in {headlines_file}")
        
        # Create a mapping from direct variable names to standard names
        column_mapping = {
            'GDP': 'Gross Domestic Product',
            'CPI': 'CPI', 
            'Unemployment Rate': 'Unemployment Rate'
        }
        
        # Select and rename columns
        selected_columns = ['Simulation', 'Timestep'] + available_vars
        df = country_data[selected_columns].copy()
        
        # Rename variables to standard names
        for var_col, std_col in column_mapping.items():
            if var_col in df.columns:
                df[std_col] = df[var_col]
        
        # Rebase CPI so that the first simulation period is 100
        if 'CPI' in df.columns:
            # Find the first timestep for each simulation and rebase CPI
            for sim in df['Simulation'].unique():
                sim_mask = df['Simulation'] == sim
                sim_data = df[sim_mask]
                if len(sim_data) > 0:
                    cpi_base = sim_data['CPI'].iloc[0]
                    if not pd.isna(cpi_base) and cpi_base != 0:
                        df.loc[sim_mask, 'CPI'] = df.loc[sim_mask, 'CPI'] / cpi_base * 100
        
        # Set Timestep as index and create MultiIndex with Simulation
        df = df.set_index(['Simulation', 'Timestep'])
        
        # Convert to the same format as shallow output (MultiIndex with 'run' and 'time')
        df.index = df.index.rename(['run', 'time'])
        
        return df
        
    except Exception as e:
        st.warning(f"Error loading Monte Carlo headlines from {headlines_file}: {str(e)}")
        raise

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

def find_closest_run(runs_df, criterion, var_name='Gross Domestic Product'):
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

def get_summary_path(runs_df, path_type, var_name='Gross Domestic Product'):
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
        'Gross Domestic Product': 'Gross Domestic Product',
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
                ["Gross Domestic Product", "CPI", "Unemployment Rate"]
            )
            path_criterion = st.selectbox(
                "Select path based on:",
                ["Random Run", "Closest to Mean", "Closest to Median", "10% CI", "25% CI", "75% CI", "90% CI"]
            )
        else:
            path_variable = "Gross Domestic Product"  # Default variable for mean/median paths
            path_criterion = path_type  # Use the path type as the criterion

        alt_forecast = st.selectbox(
            "Show alternative forecasts?",
            ["None", "VAR", "Treasury Average"]
        )
        
        # Dynamic directory input based on input type
        st.header("Input Type")
        input_type = st.radio(
            "Select input type:",
            ["Monte Carlo Headlines", "Shallow Output Files"],
            help="Shallow Output: Individual CSV files per run (UK_run_*.csv). Monte Carlo Headlines: Single CSV with Simulation/Timestep columns."
        )
        
        st.markdown("""
            <div style='height: 1rem;'></div>
        """, unsafe_allow_html=True)
                
        if input_type == "Shallow Output Files":
            results_dir = st.text_input(
                "Results Directory (Shallow Output)",
                value="output-MonteCarlo-GBR/monte_carlo",
                help="Directory containing UK_run_*.csv files"
            )
            selected_file = None  # Not used for shallow output files
        else:  # Monte Carlo Headlines
            results_dir = st.text_input(
                "Results Directory (Monte Carlo Headlines)",
                value="output",
                help="Directory containing CSV file with Simulation and Timestep columns"
            )
            
            # File selection dropdown for Monte Carlo Headlines
            try:
                results_path = Path(results_dir).resolve()
                if results_path.exists():
                    # Find CSV files that look like Monte Carlo headlines
                    csv_files = list(results_path.glob("*.csv"))
                    headlines_files = []
                    
                    for file in csv_files:
                        try:
                            # Read just the header to check column structure
                            sample_df = pd.read_csv(file, nrows=0)
                            if 'Simulation' in sample_df.columns and 'Timestep' in sample_df.columns:
                                headlines_files.append(file.name)
                        except Exception:
                            continue
                    
                    if headlines_files:
                        selected_file = st.selectbox(
                            "Select Monte Carlo Headlines File:",
                            options=headlines_files,
                            help="Select the specific CSV file containing Monte Carlo results"
                        )
                    else:
                        st.warning("No valid Monte Carlo headlines files found in the directory.")
                        selected_file = None
                else:
                    st.warning("Directory does not exist.")
                    selected_file = None
            except Exception as e:
                st.warning(f"Error scanning directory: {str(e)}")
                selected_file = None
    # Define available countries
    country_options = {
        "GBR": "Great Britain",
        "FRA": "France", 
        "CAN": "Canada"
    }
    
    if input_type == "Monte Carlo Headlines":
        # Create country tabs for Monte Carlo Headlines
        country_tabs = st.tabs([f"{code} - {name}" for code, name in country_options.items()])
        
        for tab_idx, (country_code, country_name) in enumerate(country_options.items()):
            with country_tabs[tab_idx]:
                try:
                    # Load historical data for this country
                    full_hist_data = load_historical_data(country=country_code)
                    sim_start = pd.Timestamp('2014-01-01')
                    if sim_start in full_hist_data.index:
                        start_idx = full_hist_data.index.get_loc(sim_start) - 20
                        if start_idx < 0:
                            start_idx = 0
                        hist_window = full_hist_data.iloc[start_idx:]
                    else:
                        hist_window = full_hist_data
                    
                    # Load Monte Carlo results for this country
                    mc_results = load_monte_carlo_headlines(results_dir, country=country_code, filename=selected_file)
                    
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
                    
                    # Create variable tabs for this country, filtering out GDP if not available
                    available_variables = {}
                    for var_key, var_name in VARIABLES_OF_INTEREST.items():
                        if var_name in mc_results.columns:
                            available_variables[var_key] = var_name
                        elif var_name == 'Gross Domestic Product':
                            st.warning(f"No GDP data available for {country_name} ({country_code})")
                    
                    if not available_variables:
                        st.error(f"No data available for {country_name} ({country_code})")
                        continue
                    
                    tab_labels = list(available_variables.keys())
                    tabs = st.tabs(tab_labels)
                    for i, (var_key, var_name) in enumerate(available_variables.items()):
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
                    st.error(f"Error loading data for {country_name} ({country_code}): {str(e)}")
    
    else:  # Shallow Output Files
        # Original behavior for Shallow Output Files (single country, GBR)
        try:
            full_hist_data = load_historical_data(country="GBR")
            sim_start = pd.Timestamp('2014-01-01')
            if sim_start in full_hist_data.index:
                start_idx = full_hist_data.index.get_loc(sim_start) - 20
                if start_idx < 0:
                    start_idx = 0
                hist_window = full_hist_data.iloc[start_idx:]
            else:
                hist_window = full_hist_data
                
            # Load Monte Carlo results
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
            
            # Create variable tabs, filtering out GDP if not available
            available_variables = {}
            for var_key, var_name in VARIABLES_OF_INTEREST.items():
                if var_name in mc_results.columns:
                    available_variables[var_key] = var_name
                elif var_name == 'Gross Domestic Product':
                    st.warning("No GDP data available in shallow output files")
            
            if not available_variables:
                st.error("No data available in shallow output files")
            else:
                tab_labels = list(available_variables.keys())
                tabs = st.tabs(tab_labels)
                for i, (var_key, var_name) in enumerate(available_variables.items()):
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