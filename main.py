print("")
print("+++++ 1. Show all rows from the file:")
print("")


#import csv

# The path to the file:
#file_path = r"C:\Users\Paweł\PycharmProjects\Crypto\BTC2015-2023.csv"

#with open(file_path, "r") as file_csv:
#    read_csv = csv.reader(file_csv)

# File iteration by row:
#    for row in read_csv:
#        print(row)

# ------------------------------------------------------------------------------------------------------------------------------------

print("")
print("+++++ 2. Selecting data from 01.01.2016 to 31.12.2022:")
print("")


import pandas as pd

# Loading the all CSV file:
data = pd.read_csv('BTC2015-2023.csv')

# Convert 'Date' column to date type:
data['Date'] = pd.to_datetime(data['Date'])

# Selecting data from 01.01.2016 to 31.12.2022:
start_date = pd.to_datetime('2016-01-01')
end_date = pd.to_datetime('2022-12-31')
filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

# View the resulting data:
print(filtered_data)

# -------------------------------------------------------------------------------------------------------------------------------------

print("")
print("+++++ 3. Show all columns and sample rows:")
print("")


import pandas as pd

# Setting the maximum number of columns:
pd.set_option('display.max_columns', 50)

data = pd.read_csv('BTC2015-2023.csv')
print(data.head(10))

# -----------------------------------------------------------------------------------------------------------------------------------

print("")
print("+++++ 4. Quantity of all rows and columns:")
print("")


import pandas as pd

data = pd.read_csv('BTC2015-2023.csv')
rows, columns = data.shape

print("Number of rows:", rows)
print("Number of columbs:", columns)

# -----------------------------------------------------------------------------------------------------------------------------------

print("")
print("+++++ 5. DataFrame information:")
print("")


print(data.info())

# -----------------------------------------------------------------------------------------------------------------------------------

print("")
print("+++++ 6. The number of missing values in each column:")
print("")


print(data.isnull().sum())

# -----------------------------------------------------------------------------------------------------------------------------------

print("")
print("+++++ 7. The 20 rows with the biggest difference between Close and Open, and High and Low:")
print("")


import pandas as pd

data = pd.read_csv('BTC2015-2023.csv')

data['Date'] = pd.to_datetime(data['Date'])

# Selecting data from 01.01.2016 to 31.12.2022:
start_date = pd.to_datetime('2016-01-01')
end_date = pd.to_datetime('2022-12-31')
filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)].copy()

# Calculating the difference between 'Close' and 'Open':
filtered_data.loc[:, 'Close - Open'] = filtered_data['Close'] - filtered_data['Open']

# Calculating the difference between 'High' and 'Low':
filtered_data.loc[:, 'High - Low'] = filtered_data['High'] - filtered_data['Low']

# Sorting data by difference between 'Close' and 'Open':
sorted_data_open_close = filtered_data.sort_values(by='Close - Open', ascending=False)

# Sorting data by difference between 'High' and 'Low':
sorted_data_high_low = filtered_data.sort_values(by='High - Low', ascending=False)

# Selecting the 10 rows with the biggest difference between 'Close' and 'Open':
top_10_open_close = sorted_data_open_close.head(10)

# Selecting the 10 rows with the biggest difference between 'High' i 'Low':
top_10_high_low = sorted_data_high_low.head(10)

# Combine the results:
combined_data = pd.concat([top_10_open_close, top_10_high_low])

# Showing the results:
print(combined_data)

# -----------------------------------------------------------------------------------------------------------------------------------

print("")
print("+++++ 8. Summing up prices for individual days and the difference between Close and Open, and High and Low:")
print("")


import pandas as pd

data = pd.read_csv('BTC2015-2023.csv')

data['Date'] = pd.to_datetime(data['Date'])

# Selecting data from 01.01.2016 to 31.12.2022:
start_date = pd.to_datetime('2016-01-01')
end_date = pd.to_datetime('2022-12-31')
filtered_data = data.loc[(data['Date'] >= start_date) & (data['Date'] <= end_date)].copy()

# Create a new column for an individual day:
filtered_data['Day'] = filtered_data['Date'].dt.date

# Grouping data and calculating the average for individual columns:
mean_data = filtered_data.groupby('Day')[['Open', 'High', 'Low', 'Close']].mean()

# Calculate the difference between Close and Open and between High and Low:
mean_data['Close - Open'] = mean_data['Close'] - mean_data['Open']
mean_data['High - Low'] = mean_data['High'] - mean_data['Low']

print(mean_data)

# -----------------------------------------------------------------------------------------------------------------------------------

print("")
print("+++++ 9. Highest and lowest BTC price between 01.01.2016 and 31.12.2022:")
print("")


import pandas as pd

data = pd.read_csv('BTC2015-2023.csv')

data['Date'] = pd.to_datetime(data['Date'])

start_date = pd.to_datetime('2016-01-01')
end_date = pd.to_datetime('2022-12-31')
filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

# Finding the row with the highest price:
row_with_highest_price = filtered_data.loc[filtered_data['High'].idxmax()]

# Finding the row with the lowest price:
row_with_lowest_price = filtered_data.loc[filtered_data['Low'].idxmin()]

print("Row with the highest price:")
print(row_with_highest_price)
print("\nRow with lowest price:")
print(row_with_lowest_price)

# -----------------------------------------------------------------------------------------------------------------------------------

print("")
print("+++++ 10. Average monthly price from all 4 columns between 01.01.2016 and 31.12.2022:")
print("")


import pandas as pd

data = pd.read_csv('BTC2015-2023.csv')

data['Date'] = pd.to_datetime(data['Date'])

start_date = pd.to_datetime('2016-01-01')
end_date = pd.to_datetime('2022-12-31')
filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)].copy()

# Adding a column with year and month:
filtered_data['YearMonth'] = filtered_data['Date'].dt.to_period('M')

# Group data by year and month, then calculate the total for each column:
summed_data = filtered_data.groupby('YearMonth')[['Open', 'High', 'Low', 'Close']].sum()

# Calculate the number of all prices in a particular month:
summed_data['Count'] = filtered_data.groupby('YearMonth')['Open'].count()

# Average price calculation for each month:
summed_data['Average'] = summed_data.sum(axis=1) / (4 * summed_data['Count'])

print(summed_data['Average'])

# -----------------------------------------------------------------------------------------------------------------------------------

print("")
print("+++++ 11. Average annual price from all 4 columns between 01.01.2016 and 31.12.2022:")
print("")


import pandas as pd

data = pd.read_csv('BTC2015-2023.csv')

data['Date'] = pd.to_datetime(data['Date'])

start_date = pd.to_datetime('2016-01-01')
end_date = pd.to_datetime('2022-12-31')
filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)].copy()

# Adding a column with the year:
filtered_data['Year'] = filtered_data['Date'].dt.year

# Group data by year and calculate the sum for each column:
summed_data = filtered_data.groupby('Year')[['Open', 'High', 'Low', 'Close']].sum()

# Calculation of the total number of data in a given year:
summed_data['Count'] = filtered_data.groupby('Year')['Open'].count()

# Calculate the average sum for each year:
summed_data['Average'] = summed_data.sum(axis=1) / (4 * summed_data['Count'])

# Sort results in descending order:
summed_data = summed_data.sort_values('Average', ascending=False)

# Sort results by ascending year:
summed_data = summed_data.sort_index()

# Showing 7 results:
top_results = summed_data.head(7)
print(top_results['Average'])

# -----------------------------------------------------------------------------------------------------------------------------------

print("")
print("+++++ 12. Monthly price change 4 months before and after the halving that took place in July 2016 and July 2020:")
print("")


import pandas as pd

# Round numbers to two decimal places:
pd.set_option('display.float_format', lambda x: '%.2f' % x)

data = pd.read_csv('BTC2015-2023.csv')

data['Date'] = pd.to_datetime(data['Date'])

start_date = pd.to_datetime('2015-01-01')
end_date = pd.to_datetime('2022-12-31')
filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)].copy()

# Adding a column with year and month:
filtered_data['YearMonth'] = filtered_data['Date'].dt.to_period('M')

# Group data by year and month, then calculate the average for each column:
avg_data = filtered_data.groupby('YearMonth')[['Open', 'High', 'Low', 'Close']].mean()

# Calculation of the difference between successive months:
avg_data['Price Difference'] = avg_data['Close'].diff()

# Specifying time periods:
period_2016_start = pd.Period('2016-03')
period_2016_end = pd.Period('2016-11')

period_2020_start = pd.Period('2020-03')
period_2020_end = pd.Period('2020-11')

# Selecting data for specific time periods:
period_2016_data = avg_data[(avg_data.index >= period_2016_start) & (avg_data.index <= period_2016_end)]
period_2020_data = avg_data[(avg_data.index >= period_2020_start) & (avg_data.index <= period_2020_end)]

# View data and differences for specific time periods:
print("July 2016:")
print(avg_data.loc['2016-07', ['Close', 'Price Difference']])

print("July 2020:")
print(avg_data.loc['2020-07', ['Close', 'Price Difference']])

print("4 months before July 2016:")
print(period_2016_data.loc['2016-03':'2016-06', ['Open', 'Price Difference']])

print("4 months after July 2016:")
print(period_2016_data.loc['2016-08':'2016-11', ['Open', 'Price Difference']])

print("4 months before July 2020:")
print(period_2020_data.loc['2020-03':'2020-06', ['Open', 'Price Difference']])

print("4 months after July 2020:")
print(period_2020_data.loc['2020-08':'2020-11', ['Open', 'Price Difference']])

# -----------------------------------------------------------------------------------------------------------------------------------

print("")
print("+++++ 13. Standard deviation and range of the price change:")
print("")


import pandas as pd

data = pd.read_csv('BTC2015-2023.csv')

data['Date'] = pd.to_datetime(data['Date'])

start_date = pd.to_datetime('2016-01-01')
end_date = pd.to_datetime('2022-12-31')
filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)].copy()

# Calculation of daily price changes:
filtered_data['Price Change'] = filtered_data['Close'].diff()

# Calculation of volatility indices:
volatility = filtered_data['Price Change'].std()
range_high = filtered_data['High'].max() - filtered_data['Low'].min()

print("Volatility indicators:")
print("Daily standard deviation of price changes:", volatility)
print("Range of price change: the difference between the highest and lowest BTC price between 2016 and 2022:", range_high)

# -----------------------------------------------------------------------------------------------------------------------------------

print("")
print("+++++ 14. Monthly percentage change in price - ROC indicator:")
print("")


import pandas as pd

data = pd.read_csv('BTC2015-2023.csv')

data['Date'] = pd.to_datetime(data['Date'])

# Set "Date" column as index:
data.set_index('Date', inplace=True)

start_date = pd.to_datetime('2016-01-01')
end_date = pd.to_datetime('2022-12-31')
data = data.loc[(data.index >= start_date) & (data.index <= end_date)]

# Change of data to collective monthly:
monthly_data = data.resample('M').last()

# Calculation of the indicator of increment ROC:
monthly_data['ROC %'] = (monthly_data['Close'].pct_change() * 100).round(2)

print(monthly_data[['Close', 'ROC %']])

# -----------------------------------------------------------------------------------------------------------------------------------

print("")
print("+++++ 15. Checking if there are any repeating values:")
print("")

import pandas as pd

data = pd.read_csv("BTC2015-2023.csv")

# Selecting rows that contain repeats:
duplicated_rows = data[data.duplicated()]

# Display rows containing repeats:
print(duplicated_rows)

# -----------------------------------------------------------------------------------------------------------------------------------

print("")
print("+++++ 16. Indicator WPR Williams %R:")
print("")


import pandas as pd

data = pd.read_csv('BTC2015-2023.csv')

data['Date'] = pd.to_datetime(data['Date'])

start_date = pd.to_datetime('2016-01-01')
end_date = pd.to_datetime('2022-12-31')
data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

# Calculation of the average price for the day:
data_daily = data.resample('D', on='Date').mean(numeric_only=True)

# Calculation of the Williams indicator:
period = 14
data_daily['Highest High'] = data_daily['High'].rolling(window=period).max()
data_daily['Lowest Low'] = data_daily['Low'].rolling(window=period).min()
data_daily['Williams %R'] = (data_daily['Highest High'] - data_daily['Close']) / (data_daily['Highest High'] - data_daily['Lowest Low']) * -100

# Delete rows with missing values:
data_daily = data_daily.dropna()

# Restore 'Date' column as index column:
data_daily = data_daily.reset_index()

print(data_daily[['Date', 'Open', 'High', 'Low', 'Close', 'Williams %R']])

# -----------------------------------------------------------------------------------------------------------------------------------

print("")
print("+++++ 17. Williams WPR %R indicator: best days to buy and sell BTC:")
print("")


# Set the option to display all rows:
pd.set_option('display.max_rows', None)

# Filter out days where the Williams %R is between 0 and -0:
filtered_range_1 = data_daily.query('(0 <= `Williams %R` <= -5)')

# Filter out days where the Williams %R is between -95 and -100:
filtered_range_2 = data_daily.query('(-100 <= `Williams %R` <= -95)')

# Number of rows in each range:
count_range_1 = filtered_range_1.shape[0]
count_range_2 = filtered_range_2.shape[0]

# Filter the results for an indicator between -5 and 0:
filtered_5_to_0 = data_daily.loc[(data_daily['Williams %R'] > -5) & (data_daily['Williams %R'] < 0)]
count_5_to_0 = filtered_5_to_0.shape[0]

# Filter the results for an indicator between -100 and -95:
filtered_100_to_95 = data_daily.loc[(data_daily['Williams %R'] > -100) & (data_daily['Williams %R'] < -95)]
count_100_to_95 = filtered_100_to_95.shape[0]


# View results:
print("Results for the indicator between -5 and 0: days in which to buy BTC:")
print(filtered_5_to_0)
print("The number of rows for this range:", count_5_to_0)

print("\nResults for the indicator between -100 and -95: days in which to sell BTC:")
print(filtered_100_to_95)
print("The number of rows for this range:", count_100_to_95)

# -----------------------------------------------------------------------------------------------------------------------------------

print("")
print("+++++ 18. Williams WPR %R indicator: its lowest and highest value for BTC:")
print("")


# Checking the minimum and maximum value of Williams %R in the analyzed period:
min_value = data_daily['Williams %R'].min()
max_value = data_daily['Williams %R'].max()
print("Minimum value of Williams %R:", min_value)
print("Maximum value of Williams %R:", max_value)

# -----------------------------------------------------------------------------------------------------------------------------------

print("")
print("+++++ 19. Price Rate of Change PROC expressed as a percentage:")
print("")

import pandas as pd

data = pd.read_csv('BTC2015-2023.csv')

# Convert "Date" column to datetime format:
data['Date'] = pd.to_datetime(data['Date'])

# Data selection between 01.01.2016 and 31.12.2022:
start_date = pd.to_datetime('2016-01-01')
end_date = pd.to_datetime('2022-12-31')
data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

# Set "Date" column as index:
data = data.set_index('Date')

# Calculation of the average price per month:
data_monthly = data.resample('M').mean(numeric_only=True)

# Calculation of the indicator Price Rate of Change PROC:
period = 1  # dla miesiąca, można zmienić na inny okres, np. 7 dla tygodnia
data_monthly['PROC'] = data_monthly['Close'].pct_change(period) * 100

# Set the option to display all rows:
pd.set_option('display.max_rows', None)

# Results display:
print(data_monthly[['Close', 'PROC']])

# -----------------------------------------------------------------------------------------------------------------------------------

print("")
print("+++++ 20. 50 and 200-day moving average (MA) and convergence indicator (MACD):")
print("")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY

# Loading data from a CSV file and manually converting the Date column to a datetime type:
df = pd.read_csv('BTC2015-2023.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Set the "Timestamp" column as the index of the DataFrame:
df.set_index('Date', inplace=True)

# Upewnienie się, że indeks czasowy jest posortowany rosnąco
df.sort_index(inplace=True)

# Selecting the appropriate data range from 01.01.2016 to 31.12.2022:
start_date = '2016-01-01'
end_date = '2022-12-31'
df_selected = df[start_date:end_date]

# Moving average (MA) calculation for 50 and 200 periods:
ma_72000 = df_selected['Close'].rolling(window=72000).mean()
ma_288000 = df_selected['Close'].rolling(window=288000).mean()

# Calculation of difference between MA50 and MA200 (MACD indicator):
macd = ma_72000 - ma_288000

# Visualization of results:
fig, axs = plt.subplots(2, 1, figsize=(12, 12))

axs[0].plot(df_selected.index, df_selected['Close'], label='Cena BTC', color='blue')
axs[0].plot(df_selected.index, ma_72000, label='MA(72000)', color='orange')
axs[0].plot(df_selected.index, ma_288000, label='MA(288000)', color='green')
axs[0].legend(loc='upper left')
axs[0].set_ylabel('Price BTC')
axs[0].set_title('Moving average for BTC price (01.01.2016 - 31.12.2022)')

axs[1].plot(df_selected.index, macd, label='MACD', color='red')
axs[1].axhline(y=0, color='gray', linestyle='--')                                              # Adding a zero line
axs[1].legend(loc='upper left')
axs[1].set_xlabel('Data')
axs[1].set_ylabel('MACD')
axs[1].set_title('MACD indicator for BTC price (01.01.2016 - 31.12.2022)')

# X axis formatting:
for ax in axs:
    ax.xaxis.set_major_locator(WeekdayLocator(byweekday=MONDAY))
    ax.xaxis.set_minor_locator(DayLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    ax.autoscale_view()

plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------------------------------------------------------------

print("")
print("+++++ 21. RSI indicator for the day:")
print("")


import pandas as pd

df = pd.read_csv('BTC2015-2023.csv')

# Convert "Date" column to date format:
df['Date'] = pd.to_datetime(df['Date'])

# Set "Date" column as index:
df.set_index('Date', inplace=True)

# Filter data in a specific date range:
start_date = '2016-01-01'
end_date = '2022-12-31'
df_filtered = df.loc[start_date:end_date]

# Calculation of the average price per day:
daily_average_price = df_filtered['Close'].resample('D').mean()

# Definition of the function that calculates the RSI indicator:
def calculate_rsi(data, window):
    close_prices = data['Close']
    delta = close_prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# RSI calculation for data from 01.01.2016 to 31.12.2022:
rsi_window = 14
rsi = calculate_rsi(df_filtered, rsi_window)

# Reduction of results to daily values:
daily_rsi = rsi.resample('D').last()

# Combination of results into one DataFrame:
result_df = pd.concat([daily_average_price, daily_rsi], axis=1)
result_df.columns = ['Close', 'RSI']

print(result_df)

# -----------------------------------------------------------------------------------------------------------------------------------

print("")
print("+++++ 22. RSI indicator for a given month plus RSI chart with BTC price:")
print("")


import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('BTC2015-2023.csv')

# Convert "Date" column to date format:
df['Date'] = pd.to_datetime(df['Date'])

# Set "Date" column as index:
df.set_index('Date', inplace=True)

start_date = '2016-01-01'
end_date = '2022-12-31'
df_filtered = df.loc[start_date:end_date]

# Calculation of the monthly average price:
monthly_average_price = df_filtered['Close'].resample('M').mean()

# Definition of the function that calculates the RSI indicator:
def calculate_rsi(data, window):
    close_prices = data['Close']
    delta = close_prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# RSI calculation for data from 01.01.2016 to 31.12.2022:
rsi_window = 14
rsi = calculate_rsi(df_filtered, rsi_window)

# Reduction of results to monthly values:
monthly_rsi = rsi.resample('M').last()

# Combination of results into one DataFrame:
result_df = pd.concat([monthly_average_price, monthly_rsi], axis=1)
result_df.columns = ['Close', 'RSI']

# Viewing results:
print(result_df)

# Chart for RSI indicator and BTC price:
plt.figure(figsize=(12, 6))
ax1 = plt.gca()

# RSI indicator chart:
ax1.plot(result_df.index, result_df['RSI'], label='RSI')
ax1.axhline(70, linestyle='--', color='r', label='RSI = 70')               # Addition of dashed line for value 70
ax1.axhline(30, linestyle='--', color='g', label='RSI = 30')               # Addition of dashed line for value 30
ax1.set_xlabel('Date')
ax1.set_ylabel('RSI')
ax1.set_title('Relative Strength Index (RSI)')
ax1.legend()
ax1.grid(True)

# BTC price chart on the extra right axis:
ax2 = ax1.twinx()
ax2.plot(df_filtered.index, df_filtered['Close'], label='BTC Price', color='purple')
ax2.set_ylabel('BTC Price', color='purple')
ax2.legend(loc='upper left')
ax2.tick_params(axis='y', labelcolor='purple')

plt.show()

# -----------------------------------------------------------------------------------------------------------------------------------

print("")
print("+++++ 23. Average daily BTC price from 2016 to 2022:")
print("")


import pandas as pd

# Loading data from a CSV file with appropriate encoding:
df = pd.read_csv('BTC2015-2023.csv', encoding='utf-8')

df['Date'] = pd.to_datetime(df['Date'])

start_date = pd.to_datetime('2016-01-01')
end_date = pd.to_datetime('2022-12-31')
filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()

# Converting "Open", "High", "Low", "Close" columns to numeric, and changing invalid values to NaN:
filtered_df['Open'] = pd.to_numeric(filtered_df['Open'], errors='coerce')
filtered_df['High'] = pd.to_numeric(filtered_df['High'], errors='coerce')
filtered_df['Low'] = pd.to_numeric(filtered_df['Low'], errors='coerce')
filtered_df['Close'] = pd.to_numeric(filtered_df['Close'], errors='coerce')

# Removal of rows containing NaN, that is rows containing false values:
filtered_df.dropna(inplace=True)

# Grouping by date and average calculation for "Open", "High", "Low" and "Close" columns:
df_grouped = filtered_df.groupby(filtered_df['Date'].dt.date)[['Open', 'High', 'Low', 'Close']].mean().reset_index()

# Create a new column with the average price:
df_grouped['Average'] = df_grouped[['Open', 'High', 'Low', 'Close']].mean(axis=1)

# Result display:
print(df_grouped[['Date', 'Average']])

# -----------------------------------------------------------------------------------------------------------------------------------

print("")
print("+++++ 24. The most positive and negative events for a specific year between 2016 and 2022:")
print("")


import pandas as pd

df = pd.read_csv('BTC2015-2023.csv', encoding='utf-8')

df['Date'] = pd.to_datetime(df['Date'])

start_date = pd.to_datetime('2016-01-01')
end_date = pd.to_datetime('2022-12-31')
filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()

filtered_df['Open'] = pd.to_numeric(filtered_df['Open'], errors='coerce')
filtered_df['High'] = pd.to_numeric(filtered_df['High'], errors='coerce')
filtered_df['Low'] = pd.to_numeric(filtered_df['Low'], errors='coerce')
filtered_df['Close'] = pd.to_numeric(filtered_df['Close'], errors='coerce')

filtered_df.dropna(inplace=True)

# Grouping by date and average calculation for "Open", "High", "Low" and "Close" columns:
df_grouped = filtered_df.groupby(filtered_df['Date'].dt.date)[['Open', 'High', 'Low', 'Close']].mean().reset_index()

# Create a new column with the average price:
df_grouped['Average'] = df_grouped[['Open', 'High', 'Low', 'Close']].mean(axis=1)

# List of 14 given dates:
dates_to_filter = ['2016-07-31', '2017-09-02', '2018-01-15', '2019-02-03', '2020-10-22',
                   '2021-05-16', '2022-11-07', '2016-08-22', '2017-12-16', '2018-01-13',
                   '2019-09-21', '2020-10-19', '2021-02-06', '2022-09-13']

# Function for displaying results with the possibility of adding a header:
def print_filtered_dates(target_date, header=None):
    df_grouped['Date'] = pd.to_datetime(df_grouped['Date'])                                # Date conversion to Timestamp type.
    filtered_dates = df_grouped[df_grouped['Date'] >= target_date].head(10)
    if header:
        print(header)
    print(filtered_dates[['Date', 'Average']].to_string(index=False))
    print()                                                                               # Adding empty line for better readability of results.

# List of names for each group of data:
headers = ['Bitfinex exchange hacked and 120,000 BTC stolen. Day 2016-08-02:',
           'Outlawing ICO (Initial Coin Offering) in China. Day: 2017-09-04::',
           'The controversy over BitConnect and its recognition as a Ponzi scheme. Day: 2018-01-17:',
           'Controversy around the QuadrigaCX crypto exchange and loss of access to cryptocurrency wallets. Day: 2019-02-05:',
           'Government information that cryptocurrencies are illegal in China. Day: 2020-10-24:',
           'All financial institutions and payment companies in China are banned from cryptocurrency transactions. Day: 2021-05-18:',
           'Fall of the FTX exchange. Day: 2022-11-09:',
           'Implementation of Segregated Witness (SegWit) technology for the Bitcoin BTC cryptocurrency. Day: 2016-08-24:',
           'Introduction of BTC futures on the CME Group exchange. Day: 2017-12-18:',
           'Official launch of the Lightning Network protocol for BTC. Day: 2018-01-15:',
           'Official launch of the Bakkt trading platform by the Intercontinental Exchange (ICE). Day: 2019-09-23:',
           'Announcement by PayPal about the introduction of cryptocurrency support on its payment platform. Day: 2020-10-21:',
           'Tesla bought $1.5 billion worth of BTC and plans to accept it as a form of payment for its products. Day: 2021-02-08:',
           'The Merge for Ethereum, this is the transition from proof-of-work to proof-of-stake. Day: 2022-09-15:']


# Display 5 consecutive dates and prices for each of the 14 given dates, along with the corresponding names for the data groups:
for i, date_str in enumerate(dates_to_filter):
    target_date = pd.Timestamp(date_str)
    header = headers[i % len(headers)]                                                                 # Choosing the right name for your data group.
    print_filtered_dates(target_date, header)

# -----------------------------------------------------------------------------------------------------------------------------------

print("")
print("+++++ 25. Price correlation between BTC and S&P500 between 2016 and 2022, with regression curve and trend line:")
print("")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_btc = pd.read_csv('BTC2015-2023.csv', encoding='utf-8')

df_btc['Date'] = pd.to_datetime(df_btc['Date'])

start_date = pd.to_datetime('2016-01-01')
end_date = pd.to_datetime('2022-12-31')
filtered_df_btc = df_btc[(df_btc['Date'] >= start_date) & (df_btc['Date'] <= end_date)].copy()

# Converting "Open", "High", "Low", "Close" columns to numeric, and changing invalid values to NaN:
filtered_df_btc['Open'] = pd.to_numeric(filtered_df_btc['Open'], errors='coerce')
filtered_df_btc['High'] = pd.to_numeric(filtered_df_btc['High'], errors='coerce')
filtered_df_btc['Low'] = pd.to_numeric(filtered_df_btc['Low'], errors='coerce')
filtered_df_btc['Close'] = pd.to_numeric(filtered_df_btc['Close'], errors='coerce')

# Drop rows with NaN (invalid values):
filtered_df_btc.dropna(inplace=True)

# Calculate daily average price of BTC:
df_grouped_btc = filtered_df_btc.groupby(filtered_df_btc['Date'].dt.date)[['Open', 'High', 'Low', 'Close']].mean().reset_index()
df_grouped_btc['Average'] = df_grouped_btc[['Open', 'High', 'Low', 'Close']].mean(axis=1)

# Read S&P500 data:
df_sp500 = pd.read_csv('SP500-2013-2023.csv', encoding='utf-8')

# Convert "Date" column to datetime:
df_sp500['Date'] = pd.to_datetime(df_sp500['Date'])

# Filter S&P500 data between 01.01.2016 and 31.12.2022:
filtered_df_sp500 = df_sp500[(df_sp500['Date'] >= start_date) & (df_sp500['Date'] <= end_date)].copy()

# Calculate daily average price of S&P500:
df_grouped_sp500 = filtered_df_sp500.groupby(filtered_df_sp500['Date'].dt.date)['S&P500'].mean().reset_index()

# Combining BTC and S&P500 price data based on the "Date" column:
merged_df = pd.merge(df_grouped_btc[['Date', 'Average']], df_grouped_sp500, on='Date', suffixes=('_BTC', '_S&P500'))

# Calculate correlation between BTC and S&P500:
correlation = merged_df['Average'].corr(merged_df['S&P500'])

# Calculation of the regression curve (second degree polynomial):
coefficients = np.polyfit(merged_df['Average'], merged_df['S&P500'], 2)
polynomial = np.poly1d(coefficients)

# Calculation of a linear trendline:
trend_line_coefficients = np.polyfit(merged_df['Average'], merged_df['S&P500'], 1)
trend_line_polynomial = np.poly1d(trend_line_coefficients)

# Draw a graph with points, a regression curve, and a trendline:
plt.figure(figsize=(10, 6))
plt.scatter(merged_df['Average'], merged_df['S&P500'], alpha=0.7, label='Data Points')
plt.plot(merged_df['Average'], trend_line_polynomial(merged_df['Average']), color='green', linestyle='dashed', label='Trend Line')
plt.plot(merged_df['Average'], polynomial(merged_df['Average']), color='red', linestyle='dotted', label='Regression Curve')
plt.xlabel('BTC Daily Average Price')
plt.ylabel('S&P500 Daily Average Price')
plt.title(f"Correlation between BTC and S&P500\nCorrelation Coefficient: {correlation:.2f}")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------------------------------------------------------------------------------------------------------------

print("")
print("+++++ 26. Average BTC price between 2016 and 2022:")
print("")


import numpy as np

# Loading data from a CSV file into a numpy library table:
data = np.genfromtxt('BTC2015-2023.csv', delimiter=',', skip_header=1, usecols=(3,), dtype=float)

# Data filtering only for the period 01.01.2016 - 31.12.2022:
start_date = np.datetime64('2016-01-01')
end_date = np.datetime64('2022-12-31')
dates = np.genfromtxt('BTC2015-2023.csv', delimiter=',', skip_header=1, usecols=(1,), dtype=str)
dates = np.array([np.datetime64(date) for date in dates])
mask = (dates >= start_date) & (dates <= end_date)
filtered_prices = data[mask]

# Calculation of the average BTC price:
average_price = np.mean(filtered_prices)
print("Average price of BTC in the period 01.01.2016 - 31.12.2022:", average_price)

# -----------------------------------------------------------------------------------------------------------------------------------

print("")
print("+++++ 27. Median BTC price between 2016 and 2022:")
print("")


import numpy as np

# Load data from CSV file into numpy table, skipping first row with headers:
data = np.genfromtxt('BTC2015-2023.csv', delimiter=',', skip_header=1, usecols=(3,), dtype=float)

# Data filtering only for the period 01.01.2016 - 31.12.2022:
start_date = np.datetime64('2016-01-01')
end_date = np.datetime64('2022-12-31')
dates = np.genfromtxt('BTC2015-2023.csv', delimiter=',', skip_header=1, usecols=(1,), dtype=str)
dates = np.array([np.datetime64(date) for date in dates])
mask = (dates >= start_date) & (dates <= end_date)
filtered_prices = data[mask]

# Calculation of median BTC prices:
median_price = np.median(filtered_prices)
print("Median BTC price:", median_price)

# -----------------------------------------------------------------------------------------------------------------------------------

print("")
print("+++++ 28. Correlation matrix between 4 columns: Open, High, Low, Close:")
print("")


import pandas as pd

# Loading data from a CSV file with appropriate encoding:
df = pd.read_csv('BTC2015-2023.csv', encoding='utf-8')

# Selecting the appropriate columns (Open, High, Low, Close):
selected_columns = df[['Open', 'High', 'Low', 'Close']]

# Calculation of the correlation matrix:
correlation_matrix = selected_columns.corr()

print(correlation_matrix)

# -----------------------------------------------------------------------------------------------------------------------------------

print("")
print("+++++ 29. The number of days in which the BTC price increased and in how many days it decreased:")
print("")


import pandas as pd
import numpy as np

# Loading data from a CSV file with appropriate encoding:
df = pd.read_csv('BTC2015-2023.csv', encoding='utf-8')

# Changing the "Date" column to a datetime object:
df['Date'] = pd.to_datetime(df['Date'])

# Selecting data between 01.01.2016 and 31.12.2022:
start_date = pd.to_datetime('2016-01-01')
end_date = pd.to_datetime('2022-12-31')
filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()

# Convert "Open", "High", "Low", "Close" columns to numeric, and replace invalid values with NaN:
filtered_df['Open'] = pd.to_numeric(filtered_df['Open'], errors='coerce')
filtered_df['High'] = pd.to_numeric(filtered_df['High'], errors='coerce')
filtered_df['Low'] = pd.to_numeric(filtered_df['Low'], errors='coerce')
filtered_df['Close'] = pd.to_numeric(filtered_df['Close'], errors='coerce')

# Removal of rows containing NaN this is invalid values:
filtered_df.dropna(inplace=True)

# Group by date and average price calculation for "Open", "High", "Low" and "Close" columns:
df_grouped = filtered_df.groupby(filtered_df['Date'].dt.date)[['Open', 'High', 'Low', 'Close']].mean().reset_index()

# Create a new average price column:
df_grouped['Average'] = df_grouped[['Open', 'High', 'Low', 'Close']].mean(axis=1)

# Calculation of the difference between successive average prices:
average_price_diff = np.diff(df_grouped['Average'])

# Number of days when the average price increased (positive difference):
days_rise = np.count_nonzero(average_price_diff > 0)

# Number of days when the average price decreased (negative difference):
days_fall = np.count_nonzero(average_price_diff < 0)

print("In so many days, the price has increased:", days_rise)
print("In so many days, the price has decreased:", days_fall)

# -----------------------------------------------------------------------------------------------------------------------------------

print("")
print("+++++ 30. BTC price histogram:")
print("")


import pandas as pd
import matplotlib.pyplot as plt

# Loading data:
data = pd.read_csv('BTC2015-2023.csv')

# Convert 'Date' column to date type:
data['Date'] = pd.to_datetime(data['Date'])

# Selecting data between 01.01.2016 and 31.12.2022:
start_date = pd.to_datetime('2016-01-01')
end_date = pd.to_datetime('2022-12-31')
filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)].copy()

# Creating a BTC price histogram:
plt.figure(figsize=(10, 6))
plt.hist(filtered_data['Close'], bins=50, color='skyblue')

# Graph configuration:
plt.title('Histogram ceny BTC')
plt.xlabel('Cena BTC')
plt.ylabel('Liczba wystąpień')

# Graph display:
plt.show()

# -----------------------------------------------------------------------------------------------------------------------------------

print("")
print("+++++ 31. Breakdown of data for BTC from 2016 to 2022, into a training and test set:")
print("")


import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('BTC2015-2023.csv', encoding='utf-8')

# Replace the "Date" column with a datetime object:
df['Date'] = pd.to_datetime(df['Date'])

# Selecting data between 01.01.2016 and 31.12.2022:
start_date = pd.to_datetime('2016-01-01')
end_date = pd.to_datetime('2022-12-31')
filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()

# Convert "Open", "High", "Low", "Close" columns to numeric, and replace invalid values with NaN:
filtered_df['Open'] = pd.to_numeric(filtered_df['Open'], errors='coerce')
filtered_df['High'] = pd.to_numeric(filtered_df['High'], errors='coerce')
filtered_df['Low'] = pd.to_numeric(filtered_df['Low'], errors='coerce')
filtered_df['Close'] = pd.to_numeric(filtered_df['Close'], errors='coerce')

# Removal of rows containing NaN this is invalid values:
filtered_df.dropna(inplace=True)

# Group by date and average price calculation for "Open", "High", "Low" and "Close" columns:
df_grouped = filtered_df.groupby(filtered_df['Date'].dt.date)[['Open', 'High', 'Low', 'Close']].mean().reset_index()

# Create a new average price column:
df_grouped['Average'] = df_grouped[['Open', 'High', 'Low', 'Close']].mean(axis=1)

# Result display:
print(df_grouped[['Date', 'Average']])

# Breakdown into sets: training 70% and test 30%:
train_data, test_data = train_test_split(df_grouped, test_size=0.2, random_state=42)

# View set sizes:
print("Training set size:", len(train_data))
print("Test set size:", len(test_data))

# -----------------------------------------------------------------------------------------------------------------------------------

print("")
print("+++++ 32. Training and evaluation of ML models. Regression models predict closing prices at the end of the day. Breakdown of data on 20% nad 80%:")
print("")


from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Breakdown of data into characteristics (X) and labels (y):
X_train = train_data[['Open', 'High', 'Low', 'Close']]
y_train = train_data['Close']

X_test = test_data[['Open', 'High', 'Low', 'Close']]
y_test = test_data['Close']

# Linear regression model:
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# Evaluate the quality of the linear regression model:
print("Results for the linear regression model:")
print("Mean Squared Error:", mean_squared_error(y_test, lr_pred))
print("R2 Score:", r2_score(y_test, lr_pred))

# Ridge regression model:
ridge_model = Ridge(alpha=1.0)                                              # Alpha is a hyperparameter that controls regularization
ridge_model.fit(X_train, y_train)
ridge_pred = ridge_model.predict(X_test)

# Evaluate the quality of the ridge regression model:
print("\nResults for the ridge regression model:")
print("Mean Squared Error:", mean_squared_error(y_test, ridge_pred))
print("R2 Score:", r2_score(y_test, ridge_pred))

# Random forest regression model:
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Evaluate the quality of a random forest regression model:
print("\nResults for a random forest regression model:")
print("Mean Squared Error:", mean_squared_error(y_test, rf_pred))
print("R2 Score:", r2_score(y_test, rf_pred))

# Comparison of predicted prices with actual data (first 10 values):
comparison_df = pd.DataFrame({'Actual': y_test[:10], 'Predicted Linear': lr_pred[:10],
                              'Predicted Ridge': ridge_pred[:10], 'Predicted RF': rf_pred[:10]})
print("\nComparison of predicted prices with actual data:")
print(comparison_df)

# -----------------------------------------------------------------------------------------------------------------------------------

print("")
print("+++++ 33. Predicting BTC prices from historical data using regression models: linear, ridge and random forest regression - 4 prices:")
print("")


import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

# Wczytanie danych
df = pd.read_csv('BTC2015-2023.csv', encoding='utf-8')
df['Date'] = pd.to_datetime(df['Date'])

# Przygotowanie danych treningowych dla konkretnych dni w latach 2016-2022
train_dates = pd.date_range(start='2016-01-01', end='2022-12-31')
train_data = df[df['Date'].isin(train_dates)].copy()

# Utworzenie cech i etykiet dla danych treningowych
X_train = train_data[['Open', 'High', 'Low', 'Close']]
y_train = train_data['Close']

# Inicjalizacja modeli
lr_model = LinearRegression()
ridge_model = Ridge()
rf_model = RandomForestRegressor()

# Trenowanie modeli na danych treningowych
lr_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Przygotowanie danych testowych dla konkretnych dni w 2023 roku
test_dates_2023 = pd.date_range(start='2023-01-01', end='2023-01-10')
test_data_2023 = df[df['Date'].isin(test_dates_2023)].copy()

# Utworzenie cech dla danych testowych 2023
X_test_2023 = test_data_2023[['Open', 'High', 'Low', 'Close']]

# Predykcje dla roku 2023
lr_pred_2023 = lr_model.predict(X_test_2023)
ridge_pred_2023 = ridge_model.predict(X_test_2023)
rf_pred_2023 = rf_model.predict(X_test_2023)

# Przygotowanie danych do wyświetlenia w formie DataFrame
results_df = pd.DataFrame({
    'Date': test_data_2023['Date'],
    'Predicted Average (Linear Regression)': lr_pred_2023,
    'Predicted Average (Ridge Regression)': ridge_pred_2023,
    'Predicted Average (Random Forest Regression)': rf_pred_2023
})

# Wyświetlenie wyników dla każdego modelu w kolejności daty
results_df_sorted = results_df.sort_values(by='Date')
print("Predicted prices for the year 2023 using Linear Regression:")
print(results_df_sorted[['Date', 'Predicted Average (Linear Regression)']])

print("\nPredicted prices for the year 2023 using Ridge Regression:")
print(results_df_sorted[['Date', 'Predicted Average (Ridge Regression)']])

print("\nPredicted prices for the year 2023 using Random Forest Regression:")
print(results_df_sorted[['Date', 'Predicted Average (Random Forest Regression)']])

# -----------------------------------------------------------------------------------------------------------------------------------

print("")
print("+++++ 34. Training and evaluation of ML models. Regression models predict closing prices at the end of the day (average price of the day and columns with open, high, low, close prices as a feature):")
print("")


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Loading data from a CSV file with appropriate encoding:
df = pd.read_csv('BTC2015-2023.csv', encoding='utf-8')

# Changing the "Date" column to a datetime object:
df['Date'] = pd.to_datetime(df['Date'])

# Selecting data between 01.01.2016 and 31.12.2022:
start_date = pd.to_datetime('2016-01-01')
end_date = pd.to_datetime('2022-12-31')
filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()

# Convert "Open", "High", "Low", "Close" columns to numeric, and replace invalid values with NaN:
filtered_df['Open'] = pd.to_numeric(filtered_df['Open'], errors='coerce')
filtered_df['High'] = pd.to_numeric(filtered_df['High'], errors='coerce')
filtered_df['Low'] = pd.to_numeric(filtered_df['Low'], errors='coerce')
filtered_df['Close'] = pd.to_numeric(filtered_df['Close'], errors='coerce')

# Removal of rows containing NaN this is invalid values:
filtered_df.dropna(inplace=True)

# Group by date and average price calculation for "Open", "High", "Low" and "Close" columns:
df_grouped = filtered_df.groupby(filtered_df['Date'].dt.date)[['Open', 'High', 'Low', 'Close']].mean().reset_index()

# Create a new average price column:
df_grouped['Average'] = df_grouped[['Open', 'High', 'Low', 'Close']].mean(axis=1)

# Breakdown into sets: training (70%) and test (30%):
train_data, test_data = train_test_split(df_grouped, test_size=0.2, random_state=42)

# Preparation of data for modeling:
X_train = train_data.index.values.reshape(-1, 1)
y_train = train_data['Average'].values
X_test = test_data.index.values.reshape(-1, 1)
y_test = test_data['Average'].values

# Models initialization:
linear_reg_model = LinearRegression()
ridge_reg_model = Ridge()
random_forest_model = RandomForestRegressor()

# Training models:
linear_reg_model.fit(X_train, y_train)
ridge_reg_model.fit(X_train, y_train)
random_forest_model.fit(X_train, y_train)

# Predictions on test set:
y_pred_linear = linear_reg_model.predict(X_test)
y_pred_ridge = ridge_reg_model.predict(X_test)
y_pred_rf = random_forest_model.predict(X_test)

# Errors calculation:
mse_linear = mean_squared_error(y_test, y_pred_linear)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
mse_rf = mean_squared_error(y_test, y_pred_rf)

r2_linear = r2_score(y_test, y_pred_linear)
r2_ridge = r2_score(y_test, y_pred_ridge)
r2_rf = r2_score(y_test, y_pred_rf)

# Wyświetlenie wyników
print("Results for the linear regression model:")
print("Mean Squared Error:", mse_linear)
print("R2 Score:", r2_linear)

print("\nResults for the ridge regression model:")
print("Mean Squared Error:", mse_ridge)
print("R2 Score:", r2_ridge)

print("\nResults for a random forest regression model:")
print("Mean Squared Error:", mse_rf)
print("R2 Score:", r2_rf)

# Comparison of predicted prices with actual data (first 10 values):
comparison_df = pd.DataFrame({'Actual': y_test[:10], 'Predicted Linear': y_pred_linear[:10],
                              'Predicted Ridge': y_pred_ridge[:10], 'Predicted RF': y_pred_rf[:10]})
print("\nComparison of predicted prices with actual data:")
print(comparison_df)

# -----------------------------------------------------------------------------------------------------------------------------------

print("")
print("+++++ 35. BTC price prediction based on historical data using regression models: linear, ridge and random forest regression - based on average BTC prices for a given day - 5 prices:")
print("")


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import numpy as np

# Loading data from a CSV file with appropriate encoding:
df = pd.read_csv('BTC2015-2023.csv', encoding='utf-8')

# Changing the "Date" column to a datetime object:
df['Date'] = pd.to_datetime(df['Date'])

# Selecting data between 01.01.2016 and 31.12.2022:
start_date = pd.to_datetime('2016-01-01')
end_date = pd.to_datetime('2022-12-31')
filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()

# Convert "Open", "High", "Low", "Close" columns to numeric, and replace invalid values with NaN:
filtered_df['Open'] = pd.to_numeric(filtered_df['Open'], errors='coerce')
filtered_df['High'] = pd.to_numeric(filtered_df['High'], errors='coerce')
filtered_df['Low'] = pd.to_numeric(filtered_df['Low'], errors='coerce')
filtered_df['Close'] = pd.to_numeric(filtered_df['Close'], errors='coerce')

# Removal of rows containing NaN this is invalid values:
filtered_df.dropna(inplace=True)

# Group by date and average price calculation for "Open", "High", "Low" and "Close" columns:
df_grouped = filtered_df.groupby(filtered_df['Date'].dt.date)[['Open', 'High', 'Low', 'Close']].mean().reset_index()

# Preparation of data for modeling:
X = df_grouped.index.values.reshape(-1, 1)
y = df_grouped['Close'].values

# Initialization and training of a linear regression model:
model_linear = LinearRegression()
model_linear.fit(X, y)

# Initialization and training of a ridge regression model:
model_ridge = Ridge()
model_ridge.fit(X, y)

# Initialization and training of a random forest regression model:
model_random_forest = RandomForestRegressor()
model_random_forest.fit(X, y)

# Predicting future prices based on time indices:
future_dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
future_X = np.arange(len(df_grouped), len(df_grouped) + 10).reshape(-1, 1)

future_predictions_linear = model_linear.predict(future_X)
future_predictions_ridge = model_ridge.predict(future_X)
future_predictions_random_forest = model_random_forest.predict(future_X)

# View projected prices for future dates:
future_predictions_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted Average (Linear Regression)': future_predictions_linear,
    'Predicted Average (Ridge Regression)': future_predictions_ridge,
    'Predicted Average (Random Forest)': future_predictions_random_forest
})

print("Projected prices for future dates:")
print(future_predictions_df)

# -----------------------------------------------------------------------------------------------------------------------------------

print("")
print("+++++ 36. A simplified method to simulate the impact of increased volatility on BTC price predictions using linear regression:")
print("")


import pandas as pd
from sklearn.linear_model import LinearRegression

# Loading data from a CSV file with appropriate encoding:
df = pd.read_csv('BTC2015-2023.csv', encoding='utf-8')

# Changing the "Date" column to a datetime object:
df['Date'] = pd.to_datetime(df['Date'])

# Selecting data between 01.01.2016 and 31.12.2022:
start_date = pd.to_datetime('2016-01-01')
end_date = pd.to_datetime('2022-12-31')
filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()

# Convert "Open", "High", "Low", "Close" columns to numeric, and replace invalid values with NaN:
filtered_df['Open'] = pd.to_numeric(filtered_df['Open'], errors='coerce')
filtered_df['High'] = pd.to_numeric(filtered_df['High'], errors='coerce')
filtered_df['Low'] = pd.to_numeric(filtered_df['Low'], errors='coerce')
filtered_df['Close'] = pd.to_numeric(filtered_df['Close'], errors='coerce')

# Removal of rows containing NaN this is invalid values:
filtered_df.dropna(inplace=True)

# Group by date and average price calculation for "Open", "High", "Low" and "Close" columns:
df_grouped = filtered_df.groupby(filtered_df['Date'].dt.date)[['Open', 'High', 'Low', 'Close']].mean().reset_index()

# Create a new average price column:
df_grouped['Average'] = df_grouped[['Open', 'High', 'Low', 'Close']].mean(axis=1)

# Data preparation for modelling:
X = df_grouped.index.values.reshape(-1, 1)
y = df_grouped['Average'].values

# Model initialization and training:
model = LinearRegression()
model.fit(X, y)

# Simulate scenarios and understand the impact of fluctuating factors on future prices:
df_grouped['Volatility'] = df_grouped['High'] - df_grouped['Low']
df_grouped['Increased_Volatility'] = df_grouped['Volatility'] * 1.1

# Preparation of data for modeling for increased variability:
X_volatility = df_grouped['Increased_Volatility'].values.reshape(-1, 1)

# Initialization and training of the model with new data with changed volatility:
model_with_volatility = LinearRegression()
model_with_volatility.fit(X_volatility, y)

# Price prediction based on changed volatility:
predictions_with_increased_volatility = model_with_volatility.predict(X_volatility)

# Adding results to a new data frame:
result_df = df_grouped[['Date', 'Average']].copy()
result_df['Predicted_with_Increased_Volatility'] = predictions_with_increased_volatility

# Results display:
print("\nEffect of changed volatility on predicted prices:")
print(result_df)