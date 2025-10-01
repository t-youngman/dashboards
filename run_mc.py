"""
Monte Carlo simulation script for macromodel

This uses tools from the macrocalib package - make sure you install this using
```python
pip install ".[calibration]"
```
"""

import os
import numpy as np
import pandas as pd
from macro_data import DataWrapper
from macromodel.configurations import CountryConfiguration, SimulationConfiguration
from macromodel.simulation import Simulation
from macrocalib.sampler.sampler import Sampler, process_runs

# Define countries to simulate
countries = ["GBR", "FRA", "CAN"]  # Add more countries as needed

# Country configurations
country_configurations = {
    "FRA": CountryConfiguration(),
    "CAN": CountryConfiguration(),
    "GBR": CountryConfiguration(),
}

#Simulation configuration
configuration = SimulationConfiguration(
    country_configurations=country_configurations,
    t_max=50, # number of timesteps
)

# 1. Prior sampler - placeholder, can be replaced to vary parameters in each run
def simple_prior_sampler(n_samples: int) -> np.ndarray:
    return np.tile([0.5, 1.0, 0.3], (n_samples, 1))

# 2. Configuration updater - placeholder, only changes random seed
def simple_configuration_updater(base_config: SimulationConfiguration, theta: np.ndarray) -> SimulationConfiguration:
    #base_config.seed = 42 #e.g. use to update random seed predictably
    return base_config

# 3. Observer - extract headlines time series for all countries
def simple_observer(simulation: Simulation) -> np.ndarray:
    """Extract headlines and return as flat array for compatibility with process_runs"""
    all_outputs = []
    for country in countries:
        headlines = simulation.get_country_headlines(country)
        # Keep the natural structure: [country1_all_timesteps, country2_all_timesteps, ...]
        all_outputs.append(headlines.values.flatten())
    return np.concatenate(all_outputs)

# 4. Create the sampler
data = DataWrapper.init_from_pickle("./data/processed_data/data.pkl")
sampler = Sampler(
    simulation_configuration=configuration,  # Use your custom config
    datawrapper=data,
    n_cores=4, # Change as appropriate to your system
    configuration_updater=simple_configuration_updater,
    observer=simple_observer
)

# 5. Run Monte Carlo simulations (only stochastic variation, no parameter variation)
n_runs_per_core = 1  # e.g. 25 runs per core
results = sampler.parallel_run(n_runs_per_core, simple_prior_sampler)

# 6. Process and save results
def save_monte_carlo_results(results, countries, t_max, output_dir="./output", variable_names=None):
    """
    Save Monte Carlo simulation results in multiple formats.
    
    Parameters
    ----------
    results : list[dict]
        Results from sampler.parallel_run()
    countries : list[str]
        List of country codes
    t_max : int
        Number of timesteps
    output_dir : str
        Directory to save results
    variable_names : list[str], optional
        Names of variables (e.g., ['GDP', 'CPI', 'Unemployment Rate']). 
        If None, will use generic names.
    """
    from datetime import datetime
    
    # Process results using the existing process_runs function
    x_tensor, theta_tensor = process_runs(results)
    
    # Convert to numpy arrays for easier manipulation
    x_array = x_tensor.numpy()
    theta_array = theta_tensor.numpy()
    
    # Get number of simulations and total variables
    n_simulations = x_array.shape[0]
    total_variables = x_array.shape[1]
    
    # Debug: Let's understand the data structure better
    # The model likely includes a zeroth timestep (initial conditions)
    actual_timesteps = t_max + 1  # Include zeroth timestep
    
    # Calculate variables per country per timestep
    n_vars_per_country_per_timestep = len(variable_names) if variable_names else 3
  
    # Create structured DataFrame for CSV output
    # Each row represents one timestep of one country in one simulation
    rows = []
    
    for sim_idx in range(n_simulations):
        # Extract data for this simulation
        sim_data = x_array[sim_idx]
        data_idx = 0
        
        for country_idx, country in enumerate(countries):
            for t in range(actual_timesteps):
                row = {
                    'Simulation': sim_idx,
                    'Timestep': t,
                    'Country': country,
                }
                
                # Add theta parameters if available
                for theta_idx in range(min(3, theta_array.shape[1])):
                    row[f'Theta_{theta_idx}'] = theta_array[sim_idx, theta_idx]
                
                # Add variables for this country and timestep
                for var_idx, var_name in enumerate(variable_names):
                    row[var_name] = sim_data[data_idx]
                    data_idx += 1
                
                rows.append(row)

    # Create DataFrame and save as CSV
    df_results = pd.DataFrame(rows)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"monte_carlo_results_{timestamp}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    df_results.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    
    return csv_path

# Save the results with proper variable names
variable_names = ["GDP", "CPI", "Unemployment Rate"]  # Adjust based on your model's output
csv_path = save_monte_carlo_results(
    results, countries, configuration.t_max, variable_names=variable_names
)


