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
SIM_START = pd.Timestamp('2019-01-01')  # default/fallback for formats without real timestamps (2019Q1)

# Each dashboard variable maps to a historic source: a national_accounts column name (str), a
# (table_name, column) pair for other exogenous_data tables, or None if loaded via
# CUSTOM_HISTORICAL_LOADERS. Grouped here for both the historic lookup and the tab layout.
VARIABLE_GROUPS = {
    'Prices': {
        'CPI': 'CPI (Value)',
        'PPI': 'PPI (Value)',
    },
    'Labour & Rates': {
        'Unemployment Rate': ('labour_stats', 'Unemployment Rate (Value)'),
        'Central Bank Policy Rate': None,  # loaded via CUSTOM_HISTORICAL_LOADERS
    },
    'Expenditure': {
        'GDP': 'GDP (Value)',
        'Gross Output': 'Gross Output (Value)',
        'Household Consumption': 'Household Consumption (Value)',
        'Household Investment': 'Household Investment (Value)',
        'Government Consumption': 'Government Consumption (Value)',
        'Capital Bought': 'Gross Fixed Capital Formation (Value)',
        'Imports': 'Imports (Value)',
        'Exports': 'Exports (Value)',
    },
    'Production & Income': {
        'Used Input Costs': 'Intermediate Consumption (Value)',
        'Taxes Paid on Production': 'Taxes less Subsidies on Production (Value)',
        'Taxes on Products': 'Taxes less Subsidies on Products (Value)',
        'Operating Surplus': 'Gross Operating Surplus and Mixed Income (Value)',
        'Wages': 'Compensation of Employees (Value)',
        'Inventory Changes': 'Changes in Inventories (Value)',
    },
}
VARIABLES_OF_INTEREST = {var: var for group in VARIABLE_GROUPS.values() for var in group}
PRICE_INDEX_VARS = {'CPI', 'PPI'}  # rebased to 100 at SIM_START in both historic and MC data
CONFIDENCE_LEVELS = {
    '90%': 0.90,
    '10%': 0.10
}


def _find_data_path(rel: Path, check) -> Path:
    """Search likely project-relative locations for a data path (file or directory)."""
    here = Path(__file__).resolve().parent
    for path in (here.parent / rel, here.parent / "inet-macro-dev" / rel, Path.cwd() / rel):
        if check(path):
            return path
    raise FileNotFoundError(f"Could not find {rel} under any expected project location.")


def _raw_data_path() -> Path:
    return _find_data_path(Path("data/raw_data"), Path.is_dir)


def _load_policy_rate() -> pd.Series:
    """Load the historic GBR central bank policy rate directly from raw BIS data, using the
    same PolicyRatesReader the macro_data package itself uses."""
    from macro_data.readers.economic_data.policy_rates import PolicyRatesReader

    raw_data = _raw_data_path()
    reader = PolicyRatesReader(
        path=raw_data / "policy_rates" / "bis_cb_policy_rates.csv",
        country_code_path=raw_data / "notation" / "wikipedia-iso-country-codes.csv",
    )
    return reader.get_policy_rates("GBR")["Policy Rate"]


CUSTOM_HISTORICAL_LOADERS = {"Central Bank Policy Rate": _load_policy_rate}


def _historic_series(exo, source):
    """Look up one variable's historic series from an ExogenousCountryData object."""
    table_name, column = source if isinstance(source, tuple) else ("national_accounts", source)
    table = getattr(exo, table_name, None)
    return table[column] if table is not None and column in table else np.nan


@st.cache_data
def load_historical_data(sim_start=SIM_START):
    """Load the full exogenous data series for historical data."""
    data = DataWrapper.init_from_pickle("data/processed_data/data.pkl")
    country_data = data.synthetic_countries['GBR']
    exo = country_data.exogenous_data

    series = {
        var: _historic_series(exo, source)
        for group in VARIABLE_GROUPS.values()
        for var, source in group.items()
        if source is not None
    }
    for var, loader in CUSTOM_HISTORICAL_LOADERS.items():
        if var in VARIABLES_OF_INTEREST:
            try:
                series[var] = loader()
            except Exception as e:
                st.warning(f"Error loading historic {var}: {e}")
                series[var] = np.nan

    df = pd.DataFrame(series)
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
        df.index = pd.date_range(start=f'{start_year}-01-01', periods=len(df), freq='QS')
    # Rebase price indices so that the first simulation period is 100
    for var in PRICE_INDEX_VARS:
        if var in df.columns and sim_start in df.index:
            df[var] = df[var] / df.loc[sim_start, var] * 100
    return df

def _gross_output_series(run_data):
    """Prefer the Gross Output column; older shallow-output files only have GDP."""
    if "Gross Output" in run_data.columns:
        return run_data["Gross Output"]
    if "GDP" in run_data.columns:
        return run_data["GDP"]
    raise KeyError("Missing Gross Output and GDP columns")


def _resolve_mc_column(run_data, var):
    """Match a dashboard variable to a column in an MC or shallow-output CSV."""
    if var in run_data.columns:
        return var
    aliases = {"Taxes on Products": "Taxes on products"}
    alias = aliases.get(var)
    if alias and alias in run_data.columns:
        return alias
    lower_cols = {str(c).lower(): c for c in run_data.columns}
    return lower_cols.get(var.lower())


def _mc_variable_series(run_data, var):
    """Extract one variable's series from a parsed MC run, handling naming quirks."""
    if var == "Gross Output":
        return _gross_output_series(run_data)
    col = _resolve_mc_column(run_data, var)
    if col is None and var == "Unemployment Rate" and "Unemployment Rate (Value)" in run_data.columns:
        col = "Unemployment Rate (Value)"
    return run_data[col] if col is not None else np.nan


def _infer_sim_start(file):
    """Infer the actual simulation start date from a var_level-format CSV's Date column.
    Returns None for the standard format, which carries no real timestamps."""
    with open(file, encoding="utf-8", errors="replace") as fh:
        first_line = fh.readline()
    if not first_line.startswith("var_level"):
        return None
    dates = pd.read_csv(file, index_col=0, header=1, usecols=[0]).drop(index="Date", errors="ignore")
    return pd.to_datetime(dates.index[0], errors="coerce") if len(dates) else None


def _parse_monte_carlo_csv(file):
    """Parse a single Monte Carlo run CSV (standard or exp*_combined_data format)."""
    with open(file, encoding="utf-8", errors="replace") as fh:
        first_line = fh.readline()

    if first_line.startswith("var_level"):
        run_data = pd.read_csv(file, index_col=0, header=1)
        run_data = run_data.drop(index="Date", errors="ignore")
        # Row positions become quarters from the run's inferred sim_start (see _infer_sim_start);
        # the real per-row dates aren't needed once that start date has been captured.
        run_data = run_data.reset_index(drop=True)
    else:
        run_data = pd.read_csv(file, index_col=0)
        if 1 in run_data.index:
            run_data = run_data.drop([1])

    if "CPI" not in run_data.columns:
        raise KeyError("Missing column CPI")

    run_data = pd.DataFrame({var: _mc_variable_series(run_data, var) for var in VARIABLES_OF_INTEREST})

    for var in PRICE_INDEX_VARS:
        if var in run_data.columns:
            run_data[var] = run_data[var] / run_data[var].iloc[0] * 100
    return run_data


@st.cache_data
def load_monte_carlo_results(results_dir, _loader_version=7):
    """Load and process Monte Carlo simulation results, and use nominal Gross Output levels.
    Discovers UK_run_*.csv, *shallow_output*.csv, and exp*_combined_data.csv files."""
    results_dir = Path(results_dir).resolve()
    files = sorted(
        {
            *results_dir.glob("UK_run_*.csv"),
            *results_dir.glob("GBR_shallow_output_*.csv"),
            *results_dir.glob("GBR_shallow_output*.csv"),
            *results_dir.glob("*shallow_output*.csv"),
            *results_dir.glob("exp*.csv"),
        }
    )
    all_runs = []
    load_errors = []
    for file in files:
        try:
            run_data = _parse_monte_carlo_csv(file)
            all_runs.append(run_data)
        except Exception as e:
            load_errors.append((file.name, str(e)))
            st.warning(f"Error loading {file}: {str(e)}")
            continue
    if not all_runs:
        detail = f"matched {len(files)} file(s)"
        if load_errors:
            detail += f", first error in {load_errors[0][0]}: {load_errors[0][1]}"
        raise ValueError(f"No simulation results found in {results_dir} ({detail})")
    combined_runs = pd.concat(all_runs, keys=range(len(all_runs)), names=['run', 'time'])
    # exp*_combined_data.csv files carry real dates; other formats don't, so fall back to SIM_START.
    combined_runs.attrs['sim_start'] = _infer_sim_start(files[0]) or SIM_START
    return combined_runs

def _base_level(historical_data, sim_start, column):
    """The historic level of `column` in the quarter before sim_start, for anchoring a growth forecast."""
    base_date = sim_start - pd.DateOffset(months=3)
    if base_date in historical_data.index:
        return historical_data.loc[base_date, column]
    return historical_data[column].iloc[-1]


def _growth_pct_to_level(var_forecast, var_name, base_val):
    """Convert an in-place %-growth column into a level series anchored at base_val."""
    if var_name not in var_forecast.columns:
        return
    level = [base_val]
    for growth in var_forecast[var_name].iloc[1:]:
        level.append(np.nan if pd.isna(growth) else level[-1] * (1 + growth / 100))
    var_forecast[var_name] = level[: len(var_forecast)]


@st.cache_data
def load_var_forecast(results_dir, historical_data=None, sim_start=SIM_START):
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
            # Map variable names based on actual columns. The Poledna et al 2023 VAR only
            # models these four series, so most dashboard variables have no VAR forecast.
            var_mapping = {
                'real_gdp_growth': 'GDP',
                'inflation': 'CPI',
                'real_consumption': 'Household Consumption',
                'real_investment': 'Capital Bought',
            }
            var_forecast = pd.DataFrame()
            for var_col, std_col in var_mapping.items():
                if var_col in pivot.columns:
                    var_forecast[std_col] = pivot[var_col]
            # Set index as quarterly dates if period is integer
            period_values = var_forecast.index.values
            if all(isinstance(p, (int, np.integer)) for p in period_values) or np.issubdtype(var_forecast.index.dtype, np.integer):
                var_forecast.index = pd.date_range(start=sim_start, periods=len(var_forecast), freq='QS')
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
            _growth_pct_to_level(var_forecast, 'CPI', base_val=100)
            # Transform growth-rate columns to levels, anchored on the quarter before sim_start
            if historical_data is not None:
                for level_var in ('GDP', 'Household Consumption', 'Capital Bought'):
                    if level_var in var_forecast.columns:
                        _growth_pct_to_level(var_forecast, level_var, _base_level(historical_data, sim_start, level_var))
            return var_forecast
        # Handle wide format (old logic)
        gbr_cols = [col for col in var_data.columns if 'GBR' in col]
        if not gbr_cols:
            return None
        var_mapping = {
            'GBR_GDP': 'GDP',
            'GBR_CPI': 'CPI',
            'GBR_UNEMP': 'Unemployment Rate'
        }
        var_forecast = pd.DataFrame()
        for var_col, std_col in var_mapping.items():
            if var_col in var_data.columns:
                var_forecast[std_col] = var_data[var_col]
        if not isinstance(var_forecast.index, pd.DatetimeIndex):
            var_forecast.index = pd.date_range(start=sim_start, periods=len(var_forecast), freq='QS')
        return var_forecast
    except Exception as e:
        st.warning(f"Error loading VAR forecast: {str(e)}")
        return None

HMT_FORECAST_FILENAME = "Database_of_Average_Forecasts_for_the_UK_Economy.xlsx"


def _treasury_forecast_path():
    rel = Path("data/raw_data/HMT") / HMT_FORECAST_FILENAME
    here = Path(__file__).resolve().parent
    for path in (here.parent / rel, here.parent / "inet-macro-dev" / rel, Path.cwd() / rel):
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Treasury forecast file not found ({HMT_FORECAST_FILENAME}). "
        f"Expected under data/raw_data/HMT relative to the project root."
    )


def _treasury_vintage_row(sheet_df, vintage_date):
    vintage_dates = pd.to_datetime(sheet_df.iloc[:, 0], errors="coerce", dayfirst=True)
    idx = (vintage_dates - vintage_date).abs().idxmin()
    if pd.isna(vintage_dates.loc[idx]):
        raise ValueError("No valid forecast vintage dates found in Treasury sheet.")
    return sheet_df.loc[idx]


def _treasury_gdp_year_value(row, year):
    for col in row.index:
        try:
            if int(float(col)) == year:
                val = pd.to_numeric(row[col], errors="coerce")
                if pd.notna(val):
                    return float(val)
        except (TypeError, ValueError):
            continue
    return None


def _treasury_cpi_year_value(row, year):
    target = f"{year}Q4"
    for col in row.index:
        if str(col).upper().replace(" ", "") == target:
            val = pd.to_numeric(row[col], errors="coerce")
            if pd.notna(val):
                return float(val)
    return None


def _annual_to_quarterly_growth(annual_pct):
    return ((1 + annual_pct / 100) ** 0.25 - 1) * 100


def load_treasury_forecast(historical_data=None, sim_start=SIM_START):
    """Load Treasury average forecasts from the HMT Excel database.

    Uses the vintage published closest to sim_start and reads annual GDP growth from
    the GDP sheet and Q4 CPI inflation from the CPI sheet. Annual rates are converted
    to quarterly growth and aligned from sim_start.
    """
    try:
        file_path = _treasury_forecast_path()
        gdp_row = _treasury_vintage_row(pd.read_excel(file_path, sheet_name="GDP"), sim_start)
        cpi_row = _treasury_vintage_row(pd.read_excel(file_path, sheet_name="CPI"), sim_start)

        annual_forecasts = []
        for year in range(sim_start.year, sim_start.year + 20):
            gdp_growth = _treasury_gdp_year_value(gdp_row, year)
            if gdp_growth is None:
                break
            annual_forecasts.append((year, gdp_growth, _treasury_cpi_year_value(cpi_row, year)))

        if not annual_forecasts:
            raise ValueError(f"No Treasury GDP forecasts found for vintage closest to {sim_start.date()}.")

        quarterly_gdp = []
        quarterly_cpi = []
        for _, gdp_growth, cpi_inflation in annual_forecasts:
            quarterly_gdp.extend([_annual_to_quarterly_growth(gdp_growth)] * 4)
            if cpi_inflation is not None:
                quarterly_cpi.extend([_annual_to_quarterly_growth(cpi_inflation)] * 4)
            else:
                quarterly_cpi.extend([np.nan] * 4)

        treasury = pd.DataFrame(
            {"GDP": quarterly_gdp, "CPI": quarterly_cpi},
            index=pd.date_range(start=sim_start, periods=len(quarterly_gdp), freq="QS"),
        )

        _growth_pct_to_level(treasury, "CPI", base_val=100)
        if historical_data is not None:
            _growth_pct_to_level(treasury, "GDP", _base_level(historical_data, sim_start, "GDP"))
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
               compare_stats=None, compare_run=None, primary_label="Baseline", compare_label="Shock", shock_start_q=0, show_shock_annotation=False,
               sim_start=SIM_START, compare_sim_start=None):
    """Create a single plot for a variable. If compare_stats/compare_run are set, add second set with distinct colors."""
    fig = go.Figure()
    forecast_dates = pd.date_range(
        start=sim_start,
        periods=len(mc_stats[var_key]),
        freq='QS'
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
        compare_dates = pd.date_range(start=compare_sim_start or sim_start, periods=n_compare, freq='QS')
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
    # Historical and simulation data share column names throughout VARIABLES_OF_INTEREST.
    hist_var = sim_var = var_name
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

def build_selected_run_pct_change_table(run_a, run_b, var_name, label_b_vs_a, sim_start=SIM_START):
    """Build a table of period-by-period percentage change (run_b vs run_a) for one variable."""
    n = min(len(run_a), len(run_b))
    if n == 0 or var_name not in run_a.columns or var_name not in run_b.columns:
        return pd.DataFrame()
    dates = pd.date_range(start=sim_start, periods=n, freq='QS')
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

def create_single_var_comparison_chart(run_a, run_b, label_a, label_b, var_name, sim_start=SIM_START):
    """Create a single chart comparing the two selected runs for one variable."""
    n = min(len(run_a), len(run_b))
    if n == 0 or var_name not in run_a.columns or var_name not in run_b.columns:
        return go.Figure()
    dates = pd.date_range(start=sim_start, periods=n, freq='QS')
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
                list(VARIABLES_OF_INTEREST.keys())
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
        mc_results = load_monte_carlo_results(results_dir)
        sim_start = mc_results.attrs.get('sim_start', SIM_START)
        full_hist_data = load_historical_data.__wrapped__(sim_start)
        if sim_start in full_hist_data.index:
            start_idx = full_hist_data.index.get_loc(sim_start) - 20
            if start_idx < 0:
                start_idx = 0
            hist_window = full_hist_data.iloc[start_idx:]
        else:
            hist_window = full_hist_data
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
        var_forecast_df = load_var_forecast(results_dir, historical_data=full_hist_data, sim_start=sim_start)
        treasury_forecast_df = load_treasury_forecast(historical_data=full_hist_data, sim_start=sim_start)
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
        compare_sim_start = None
        if compare_mode and compare_dir:
            mc_results_2 = load_monte_carlo_results(compare_dir)
            compare_sim_start = mc_results_2.attrs.get('sim_start', SIM_START)
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

        group_tabs = st.tabs(list(VARIABLE_GROUPS.keys()))
        for group_tab, group_vars in zip(group_tabs, VARIABLE_GROUPS.values()):
            with group_tab:
                var_tabs = st.tabs(list(group_vars.keys()))
                for var_tab, var_name in zip(var_tabs, group_vars.keys()):
                    with var_tab:
                        fig = create_plot(
                            hist_window, mc_stats, selected_run, var_name, var_name,
                            forecast_start=hist_window.index[-1],
                            show_var=(alt_forecast != "None"), var_forecast=alt_forecast_df, alt_label=alt_label,
                            compare_stats=mc_stats_2, compare_run=selected_run_2,
                            primary_label=primary_label, compare_label=compare_label,
                            shock_start_q=shock_start_q, show_shock_annotation=show_shock_annotation,
                            sim_start=sim_start, compare_sim_start=compare_sim_start
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
                                selected_run, selected_run_2, var_name, pct_label, sim_start=sim_start
                            )
                            if not comp_table.empty:
                                styled = comp_table.style.format({pct_label: "{:+.2f}%"}, na_rep="—")
                                st.dataframe(styled, width='stretch')
                                fig_comp = create_single_var_comparison_chart(
                                    selected_run, selected_run_2, primary_label, compare_label, var_name, sim_start=sim_start
                                )
                                st.plotly_chart(fig_comp, width='stretch')

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

if __name__ == "__main__":
    main() 