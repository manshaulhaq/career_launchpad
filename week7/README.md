# Personalized Movie Discovery Engine

This project implements a recommendation system, starting from a baseline Collaborative Filtering approach to an advanced Neural Collaborative Filtering (NCF) architecture.

## Project Structure

- `analysis.ipynb`: The core notebook documenting the data science process across 5 phases (Business KPIs, Baseline CF, NCF, API, Dashboard).
- `api.py`: A FastAPI application serving personalized Top-K recommendations.
- `dashboard.py`: A Streamlit dashboard visualizing and comparing recommendations between Baseline CF and NCF models.
- `dataset/`: Directory containing the `movies.csv` and `ratings.csv` data.

## How to Run

### 1. Interactive Dashboard
To visualize recommendation comparisons:
```bash
pip install streamlit matplotlib seaborn
streamlit run dashboard.py
```

### 2. Recommendation API
To start the FastAPI server:
```bash
pip install fastapi uvicorn pandas
uvicorn api:app --reload
```
Access the interactive API docs at: `http://localhost:8000/docs`

### 3. Notebook
Open `analysis.ipynb` in a Jupyter environment to explore the data processing, model mathematics, and training logic.

## Technologies Used
- Data & Machine Learning: `pandas`, `numpy`, `scikit-learn`, `PyTorch`
- API & UI: `FastAPI`, `Uvicorn`, `Streamlit`, `matplotlib`, `seaborn`
