import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

dataframe = None
def initialize(filename):
    global dataframe
    dataframe = pd.read_csv(filename)


def holt(indicator, days, smoothing_level, smoothing_slope, predict_date):
    global dataframe
    df_tail = dataframe.tail(days)
    y_to_train = df_tail[indicator]
    fit1 = Holt(y_to_train).fit(smoothing_level, smoothing_slope, optimized=False)
    fcast1 = fit1.forecast(predict_date).rename("Holt's linear trend")
    fit2 = Holt(y_to_train, exponential=True).fit(smoothing_level, smoothing_slope, optimized=False)
    fcast2 = fit2.forecast(predict_date).rename("Exponential trend")
    x = list({number for number in range(len(fcast1)+days)})
    #df_tail[indicator].plot(marker="o", color='green', legend=True)
    plt.plot(x[:len(df_tail)], df_tail[indicator], 'g', label='Real')
    #plt.plot(x[:len(fit1.fittedvalues)], fit1.fittedvalues, 'r', label='')
    plt.plot(x[len(df_tail):], fcast1, 'r', label='Linear Holt')
    #fit2.fittedvalues.plot(marker="o", color='red')
    #fcast2.plot(color='red', marker="o", legend=True)
    #plt.plot(x[:len(fit2.fittedvalues)], fit2.fittedvalues, 'b')
    plt.plot(x[len(df_tail):], fcast2, 'b', label='Exponential Holt')
    plt.xlabel('Days')
    plt.ylabel(indicator)
    plt.legend(loc='best')
    plt.show()


def get_holt_trend(indicator, averaging, count_of_periods, smoothing):
    global dataframe
    df_tail = dataframe.tail(averaging * count_of_periods)
    y = df_tail[indicator]

    fit1 = Holt(df_tail[indicator]).fit(smoothing_level=smoothing)
    forecast = fit1.forecast(7)
    x = list({number + 1 for number in range(len(y))})
    plt.plot(x, y, label='Real')
    x = list({number for number in range(len(y), len(y)+len(forecast))})
    plt.plot(x,forecast, label='Holt_linear')
    plt.xlabel('Days')
    plt.ylabel(indicator)
    plt.legend(loc='best')
    plt.show()

def get_brown(indicator, averaging, count_of_periods, smoothing):
    global dataframe
    df_tail = dataframe.tail(averaging * count_of_periods)
    y = df_tail[indicator]
    fit2 = SimpleExpSmoothing(y).fit(smoothing_level=smoothing, optimized=False)
    forecast = fit2.forecast(7)
    x = list({number + 1 for number in range(len(y))})
    plt.plot(x, y, 'b', label='real')
    plt.plot(x, fit2.fittedvalues, 'r', label='smoothed ({})'.format(smoothing))
    x = list({number for number in range(len(y), len(y)+len(forecast))})
    plt.plot(x, forecast, 'g', label='forecast')
    plt.xlabel('Days')
    plt.ylabel(indicator)
    plt.legend(loc='best')
    plt.show()


def get_EMA(indicator, averaging, count_of_periods):
    global dataframe
    df_tail = dataframe.tail(averaging*count_of_periods)
    y = df_tail[indicator]
    ma = df_tail[indicator].ewm(span=averaging).mean()
    x = list({number+1 for number in range(averaging*count_of_periods)})
    plt.scatter(x, y)
    plt.plot(x, ma, 'r', linestyle='solid')
    plt.legend(loc='best')
    plt.show()

def get_MA(indicator, averaging, count_of_periods):
    global dataframe
    df_tail = dataframe.tail(averaging*count_of_periods)
    print(len(df_tail))
    y = df_tail[indicator]
    ma = df_tail[indicator].rolling(averaging).mean()
    x = list({number+1 for number in range(averaging*count_of_periods)})
    plt.plot(x, y, 'b', label='Real')
    plt.plot(x, ma, 'r', linestyle='solid', label='forecast')
    plt.xlabel('Days')
    plt.ylabel(indicator)
    plt.legend(loc='best')
    plt.show()


def get_trend(indicator, averaging, count_of_periods):
    global dataframe
    df_tail = dataframe.tail(averaging*count_of_periods)
    x = []
    y = []
    counter = 0
    accumulator = 0
    for res in df_tail[indicator]:
        accumulator += res
        counter += 1
        if counter % averaging == 0:
            y.append(accumulator/averaging)
            x.append(counter)
            accumulator = 0
    plt.plot(x, y, 'b', label='Real data')
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), 'r', linestyle='solid', label='Trend')
    plt.xlabel('Days')
    plt.ylabel(indicator)
    plt.legend(loc='best')
    plt.show()


