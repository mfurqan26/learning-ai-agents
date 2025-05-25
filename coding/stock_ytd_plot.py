# filename: stock_ytd_plot.py
import yfinance as yf
import matplotlib.pyplot as plt

# Define the stocks and date range
stocks = ['NVDA', 'TLSA']
start_date = '2025-01-01'
end_date = '2025-05-25'  # today's date

# Fetch stock data
data = yf.download(stocks, start=start_date, end=end_date)

# Extract closing prices from MultiIndex columns
close_nvda = data[('Close', 'NVDA')]
close_tlsa = data[('Close', 'TLSA')]

# Calculate YTD gains
initial_price_nvda = close_nvda.iloc[0]
current_price_nvda = close_nvda.iloc[-1]
gain_nvda = (current_price_nvda - initial_price_nvda) / initial_price_nvda * 100

initial_price_tlsa = close_tlsa.iloc[0]
current_price_tlsa = close_tlsa.iloc[-1]
gain_tlsa = (current_price_tlsa - initial_price_tlsa) / initial_price_tlsa * 100

# Plotting
plt.figure(figsize=(8,6))
plt.bar(['NVDA', 'TLSA'], [gain_nvda, gain_tlsa], color=['blue', 'orange'])
plt.ylabel('YTD Gain (%)')
plt.title('Stock YTD Gain for NVDA and TLSA (2025)')
plt.savefig('ytd_stock_gains.png')
print('Plot saved as ytd_stock_gains.png')