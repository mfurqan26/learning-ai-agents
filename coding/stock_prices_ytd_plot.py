# filename: stock_prices_ytd_plot.py
import pandas as pd
from datetime import datetime
from functions import get_stock_prices, plot_stock_prices

# Define the stock symbols
stock_symbols = ['NVDA', 'TSLA']

# Set date range for YTD
start_date = '2025-01-01'
end_date = '2025-05-25'  # today's date

# Get stock prices
stock_prices = get_stock_prices(stock_symbols, start_date, end_date)

# Plot and save
plot_stock_prices(stock_prices, 'stock_prices_YTD_plot.png')