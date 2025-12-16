🚀 Retail Sales Forecasting Project (WEEK 1 Summary)
🎯 Goal

Predict the Total Daily Units Sold using a time-series model.
✅ Key Achievements
Task Category	Status	Details
Data Engineering	Complete	Aggregated daily sales, created Lag features (lag_1, lag_7), and used exogenous features (Price, Discount).
Modeling	Complete	Selected Random Forest Regressor as the final model due to its high accuracy (MAPE: 0.53%), outperforming SARIMAX.
Deployment	Complete	Model saved to rf_sales_forecast_model.pkl. Flask API (gui_app.py) successfully deployed with a web GUI for user prediction.
Delivery	Complete	Forecast comparison dashboard (final_forecast_dashboard.json) and final data saved.
⚙️ Deployment Access

The prediction service is active and accessible via a simple web interface.

    GUI Link: http://127.0.0.1:5000/ (Check your console for the exact port)

    API Endpoint (for systems): POST request to http://127.0.0.1:5000/predict
