# filename: stock_ytd_time_series.py
import yfinance as yf
import matplotlib.pyplot as plt

# Define the stocks and date range
stocks = ['NVDA', 'TLSA']
start_date = '2025-01-01'
end_date = '2025-05-25'  # today

# Fetch stock data
data = yf.download(stocks, start=start_date, end=end_date)

# Extract closing prices
close_nvda = data[('Close', 'NVDA')]
close_tlsa = data[('Close', 'TLSA')]

# Calculate YTD percentage change over time
initial_price_nvda = close_nvda.iloc[0]
ytd_percent_nvda = (close_nvda - initial_price_nvda) / initial_price_nvda * 100

initial_price_tlsa = close_tlsa.iloc[0]
ytd_percent_tlsa = (close_tlsa - initial_price_tlsa) / initial_price_tlsa * 100

# Plotting
plt.figure(figsize=(10,6))
plt.plot(ytd_percent_nvda.index, ytd_percent_nvda, label='NVDA')
plt.plot(ytd_percent_tlsa.index, ytd_percent_tlsa, label='TLSA')
plt.xlabel('Date')
plt.ylabel('YTD Gain (%)')
plt.title('YTD Stock Gain from 2025-01-01 to Today')
plt.legend()
plt.grid(True)
plt.savefig('ytd_stock_time_series.png')
print('Time series plot saved as ytd_stock_time_series.png')