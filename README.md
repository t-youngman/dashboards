# Dashboards for the INET Oxford macroeconomic agent-based model

Very much a work in progress! They all use [streamlit](https://streamlit.io/), so before you begin make sure you have that installed.

```bash
pip install streamlit
```

Clone the repo within the development folder where you usually output model runs.

```bash
    ├───dev
    │   ├───output
    │   ├───dashboards
    │   └───data
    │       ├───processed_data
    │       └───raw_data
    └───macro-main
```

Run the dashboards with the equivalent of your 'dev' folder set as the working directory using the following command:

```python
streamlit run ./dashboards/mcdash.py
```

## Monte Carlo dashboard - for visualising simulations
This dashboard displays the outputs of Monte Carlo simulations of the macromodel alongside historical data and VAR projections. Its input file are .csv files of the macro model's 'Shallow Output' and .pkl files of the processed_data for the relevant historic time series. In time I hope to update what is displayed in the shallow_output files and then update what variables are available on the dashboard.

To generate the shallow output files, add the following lines to the end of your 'run model' script

```python
# Generate and save shallow output .csv file
shallow_output = model.get_country_shallow_output("GBR")
shallow_output.to_csv("./output/GBR_shallow_output.csv")
```

## h5 dashboards - for exploring results
This dashboard finds any h5 model output files in your 'dev' equivalent folder and visualises individual variables within them. I use it for exploring and debugging model runs.


To save the h5 file, add the following line to the end of your run model script, replacing 'GBR' with the three digit country code of your choice. Remember that h5 files include agent-by-agent data so get very large as the scale gets more fine grained.

```python
#Save h5 file with agent-by-agent model output
model.save(save_dir=Path("./output/"), file_name="GBR_run.h5")
```

## GDP components dashboard - for debugging
This dashboard visualises the GDP components output that is currently available in the macro model. It shows GDP projections for the three GDP calculation approaches (output, income, expenditure) in a stacked line graph breaking down by the components of each approach. I created it for debugging if the three approaches are not adding up.

To generate the GDP components .csv file needed to run this dashboard, add the following to the end of your run model script.

```python
# Generate and save GDP debug output
gdp_components = model.get_country_gdp_components_df("GBR")
gdp_components.to_csv("./output/GBR_gdp_components.csv")
```