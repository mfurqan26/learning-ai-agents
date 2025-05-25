# filename: check_columns.py
import yfinance as yf

# Define the stocks and date range
stocks = ['NVDA', 'TLSA']
start_date = '2025-01-01'
end_date = '2025-05-25'

# Fetch stock data
data = yf.download(stocks, start=start_date, end=end_date)
print(data.columns)