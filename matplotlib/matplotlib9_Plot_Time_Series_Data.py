import pandas as pd
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from matplotlib import dates as mpl_dates

plt.style.use('seaborn')

dates = [
    datetime(2019, 5, 24),
    datetime(2019, 5, 25),
    datetime(2019, 5, 26),
    datetime(2019, 5, 27),
    datetime(2019, 5, 28),
    datetime(2019, 5, 29),
    datetime(2019, 5, 30)
]
print(dates)
'''
[datetime.datetime(2019, 5, 24, 0, 0), datetime.datetime(2019, 5, 25, 0, 0), datetime.datetime(2019, 5, 26, 0, 0), 
datetime.datetime(2019, 5, 27, 0, 0), datetime.datetime(2019, 5, 28, 0, 0), datetime.datetime(2019, 5, 29, 0, 0), 
datetime.datetime(2019, 5, 30, 0, 0)]
'''
print(dates[0]) 
'''
2019-05-24 00:00:00
'''

y = [0, 1, 3, 4, 6, 5, 7]

plt.plot_date(dates, y)

# data = pd.read_csv('data.csv')
# price_date = data['Date']
# price_close = data['Close']

# plt.title('Bitcoin Prices')
# plt.xlabel('Date')
# plt.ylabel('Closing Price')

plt.tight_layout()

plt.show()


################################## Connected each point

plt.plot_date(dates, y, linestyle="solid")

# data = pd.read_csv('data.csv')
# price_date = data['Date']
# price_close = data['Close']

# plt.title('Bitcoin Prices')
# plt.xlabel('Date')
# plt.ylabel('Closing Price')

plt.tight_layout()

plt.show()

##################################  Better in x axis label ##################################
plt.plot_date(dates, y, linestyle="solid")
plt.gcf().autofmt_xdate() # gcf: get current figure
# data = pd.read_csv('data.csv')
# price_date = data['Date']
# price_close = data['Close']

# plt.title('Bitcoin Prices')
# plt.xlabel('Date')
# plt.ylabel('Closing Price')

plt.tight_layout()

plt.show()

########################## Change Date Format ########################

plt.plot_date(dates, y, linestyle="solid")
plt.gcf().autofmt_xdate() 
date_format = mpl_dates.DateFormatter("%b, %d %Y")
plt.gca().xaxis.set_major_formatter(date_format)
# data = pd.read_csv('data.csv')
# price_date = data['Date']
# price_close = data['Close']

# plt.title('Bitcoin Prices')
# plt.xlabel('Date')
# plt.ylabel('Closing Price')

plt.tight_layout()

plt.show()


########################################



data = pd.read_csv('data4.csv')


price_date = data['Date'] # PROBLEM: read as string, not date, so we need sort by day
price_close = data['Close']

plt.plot_date(price_date, price_close, linestyle="solid")
plt.gcf().autofmt_xdate() 
date_format = mpl_dates.DateFormatter("%b, %d %Y")
plt.gca().xaxis.set_major_formatter(date_format)


plt.title('Bitcoin Prices')
plt.xlabel('Date')
plt.ylabel('Closing Price')

plt.tight_layout()

plt.show()


########################################



data = pd.read_csv('data4.csv')

data['Date']  = pd.to_datetime(data["Date"])
data.sort_values("Date", inplace=True) # inplace=True mean modify directly original data, not return a modified Dataframe

price_date = data['Date'] # PROBLEM: read as string, not date, so we need sort by day
price_close = data['Close']

plt.plot_date(price_date, price_close, linestyle="solid")
plt.gcf().autofmt_xdate() 
date_format = mpl_dates.DateFormatter("%b, %d %Y")
plt.gca().xaxis.set_major_formatter(date_format)


plt.title('Bitcoin Prices')
plt.xlabel('Date')
plt.ylabel('Closing Price')

plt.tight_layout()

plt.show()
