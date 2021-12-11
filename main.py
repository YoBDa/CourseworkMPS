# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# LINK
import csv

# https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/01-01-2021.csv


import argparse
from datetime import date, timedelta
import requests
import pandas as pd
import re, os
import numpy as np
import analyze
def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

def load_csv(date_to_load, region):
    print('Downloading {}.csv'.format(date_to_load))
    url = 'https://raw.githubusercontent.com/CSSEGISandData/' \
          'COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/' \
          '{0}.csv'.format(date_to_load)
    try:
        r = requests.get(url, allow_redirects=True)
        print('Done.')
        with open('{0}.csv'.format(date_to_load), 'wb') as csvfile:
            csvfile.write(r.content)
            print('Saved.')

    except Exception as ex:
        print('Failed to request data.\n URL: {0}\n {1}'.format(url, str(ex)))
        return
    print('Processing dataset from {0}.csv'.format(date_to_load))
    dataset = pd.read_csv('{0}.csv'.format(date_to_load))
    booleans = []
    for result in dataset.Combined_Key:
        if not re.search(region, result):
            booleans.append(False)
        else:
            booleans.append(True)
    filtered = pd.Series(booleans)
    new_dataset = dataset[filtered]
    if os.path.isfile('./{}.csv'.format(region)):
        new_dataset.to_csv('{}.csv'.format(region), mode='a', header=False, index=False)
    else:
        new_dataset.to_csv('{}.csv'.format(region), mode='w', header=True, index=False)
    print('Append filtered {}.csv to {}.csv'.format(date_to_load, region))
    os.remove('./{}.csv'.format(date_to_load))


def load_data(start_month=1,start_day=1, end_month=12, end_day=31, year=2021):
    sdate = date(year, start_month, start_day)
    fdate = date(year, end_month, end_day)
    for single_date in daterange(sdate, fdate):
        date_to_load = single_date.strftime("%m-%d-%Y")
        load_csv(date_to_load, 'Moscow, Russia')
    optimize_csv('Moscow, Russia.csv')





def optimize_csv(filename):
    dataset = pd.read_csv(filename)
    dates = []

    deaths_deltas = []
    prev_deaths = 0
    for result in dataset.Deaths:
        deaths_deltas.append(result - prev_deaths)
        prev_deaths = result
    dataset['Deaths_Delta'] = deaths_deltas

    confirmed_deltas = []
    prev_confirmed = 0
    for result in dataset.Confirmed:
        confirmed_deltas.append(result - prev_confirmed)
        prev_confirmed = result
    dataset['Confirmed_Delta'] = confirmed_deltas
    for result in dataset.Last_Update:
        dates.append(result.split()[0])
    dataset['Date'] = dates
    dataset.drop('Last_Update', axis='columns', inplace=True)
    dataset.drop('FIPS', axis='columns', inplace=True)
    dataset.drop('Admin2', axis='columns', inplace=True)
    dataset.drop('Province_State', axis='columns', inplace=True)
    dataset.drop('Country_Region', axis='columns', inplace=True)
    dataset.drop('Lat', axis='columns', inplace=True)
    dataset.drop('Long_', axis='columns', inplace=True)
    dataset.drop('Combined_Key', axis='columns', inplace=True)
    dataset.to_csv(filename, index=False)



def add_index(filename):
    df = pd.read_csv(filename)
    df_len = len(df)
    df['Number'] = np.arange(df_len) + 1
    df.to_csv(filename, index=False)

def hello():
    parser = argparse.ArgumentParser(description='Covid forecast')
    #parser.add_argument('integers', metavar='N', type=int, nargs='+', help='an integer for the accumulator')
    parser.add_argument('--load-data', dest='load', action='store_true', help='load data from github & aggregate it')

    parser.parse_args('--load-data'.split())



    args = parser.parse_args()
    print(args.accumulate(args.integers))


if __name__ == '__main__':
    # load_data(12, 8, 12, 11, 2021)
    #add_index('Moscow, Russia.csv')
    #load_csv('03-03-2021')

    analyze.initialize('Moscow, Russia.csv')
    averaging = 3
    count_of_periods = 10
    # analyze.get_trend('Deaths_Delta', averaging, count_of_periods)
    # analyze.get_MA('Confirmed_Delta', averaging, count_of_periods)
    # analyze.get_EMA('Confirmed_Delta', averaging, count_of_periods)
    # analyze.get_trend('Confirmed_Delta', averaging, count_of_periods)
    # analyze.get_brown('Confirmed_Delta', averaging, count_of_periods, 0.4)
    # analyze.get_holt_trend('Confirmed_Delta', averaging, count_of_periods, 0.3)
    analyze.holt('Confirmed_Delta', 30, 0.3, 0.1, 5)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
