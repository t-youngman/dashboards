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

## h5 dashboards - for exploring results
This dashboard finds any h5 model output files in your 'dev' equivalent folder and visualises individual variables within them. I use it for exploring and debugging model runs.

## GDP components dashboard - for debugging
This dashboard visualises the GDP components output that is currently available in the macro model. It shows GDP projections for the three GDP calculation approaches (output, income, expenditure) in a stacked line graph breaking down by the components of each approach. I created it for debugging if the three approaches are not adding up.
