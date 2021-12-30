import pandas as pd
import yfinance as yf

stockNo = "2330.TW"
start_date = '2015-01-01'
df = yf.download(stockNo, start=start_date)
df = df.reset_index()
from stocker import Stocker
stock = Stocker(stockNo, df)
stock.plot_stock()
model, model_data = stock.create_prophet_model(days=10)
stock.evaluate_prediction()
stock.changepoint_prior_analysis(changepoint_priors=[0.001, 0.05, 0.1, 0.2])
stock.changepoint_prior_scale = 0.5
stock.evaluate_prediction()
