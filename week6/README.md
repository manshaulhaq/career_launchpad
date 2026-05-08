# E-Commerce Fraud Detection Simulator 🛡️

A real-time fraud detection pipeline and dashboard built to demonstrate the end-to-end process of identifying fraudulent transactions, calculating business KPIs, and simulating a live e-commerce transaction stream.

## Features
- **Exploratory Data Analysis & Feature Engineering**: Notebook covering historical fraud patterns, missing value handling, and velocity-based feature creation.
- **Machine Learning Models**: Implementation of anomaly detection (Isolation Forest) and supervised ensemble models (Random Forest).
- **Real-Time Stream Simulation**: A Streamlit dashboard (`dashboard.py`) that processes transactions in batches to mimic a live production environment.
- **Live Business KPIs**: Dynamic tracking of Net Savings, System ROI, Fraud Recall, and Operational Costs based on simulated predictions.

## Project Structure
- `fraud_detection.ipynb`: The core Jupyter Notebook containing data preprocessing, feature engineering, and model training logic.
- `dashboard.py`: A premium Streamlit application for real-time streaming visualization and KPI tracking.

## Setup & Installation

1. Ensure the IEEE-CIS Fraud Detection dataset is located in the `dataset/` directory (specifically `train_transaction.csv` and `train_identity.csv`).
2. Install the necessary Python dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn streamlit plotly jupyter
   ```
   *(Note: Depending on your environment, you may need to use `pipx`, a virtual environment, or pass `--break-system-packages` if externally managed).*

## Running the Dashboard

To launch the real-time simulation dashboard, open your terminal and run:

```bash
streamlit run dashboard.py
```

## Usage
Once the dashboard opens in your browser, use the **⚙️ Simulation Settings** in the sidebar to:
- Adjust the **Batch Size** and **Speed** of the incoming data stream.
- Modify the **Cost per Review** to see how manual review costs impact your overall Net Savings and ROI in real-time.
- Click **Start**, **Stop**, or **Reset** to control the simulation loop.
