import plotly.io as pio

# --- 1. Define the filename ---
plotly_filename = 'final_forecast_dashboard.json'

# --- 2. Load the Plotly figure from the JSON file ---
try:
    # Use pio.read_json() to deserialize the JSON file back into a Plotly Figure object
    fig = pio.read_json(plotly_filename)
    
    print(f"Plotly figure successfully loaded from {plotly_filename}")

    # --- 3. Display the interactive figure ---
    # In a Jupyter/IPython environment, fig.show() will render the interactive chart.
    fig.show()

except FileNotFoundError:
    print(f"Error: The file {plotly_filename} was not found. Please ensure it is in the current directory.")
except Exception as e:
    print(f"An error occurred while loading the plot: {e}")
