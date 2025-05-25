# filename: download_and_plot_stock_prices_ytd.py
from datetime import datetime
from functions import get_stock_prices, plot_stock_prices

start_date = '2025-01-01'
end_date = '2025-05-25'
stock_symbols = ['NVDA', 'TSLA']

# Get the stock prices for NVDA and TSLA year-to-date
stock_prices = get_stock_prices(stock_symbols, start_date, end_date)

# Plot and save the stock prices
plot_stock_prices(stock_prices, 'stock_prices_YTD_plot.png')