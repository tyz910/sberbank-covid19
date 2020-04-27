import requests
import pandas as pd


def update_data():
    time_series_data = {
        'confirmed.csv': 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv',
        'deaths.csv': 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv',
        'confirmed_ru.csv': 'https://raw.githubusercontent.com/grwlf/COVID-19_plus_Russia/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_RU.csv',
        'deaths_ru.csv': 'https://raw.githubusercontent.com/grwlf/COVID-19_plus_Russia/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_RU.csv',
    }

    for file_name, url in time_series_data.items():
        r = requests.get(url, allow_redirects=True)
        if r.status_code == 200:
            print(file_name, 'loaded')
            with open('data/time_series/' + file_name, 'wb') as f:
                f.write(r.content)


def read_time_series(countries, filepath):
    df = pd.read_csv(filepath, na_filter=False)

    df = df.rename(columns={
        'Province_State': 'Country/Region',
        'Country_Region': 'Province/State',
    }).drop(columns=[
        'Lat', 'Long', 'Province/State', 'UID', 'iso2', 'iso3',
        'code3', 'FIPS', 'Admin2', 'Long_', 'Combined_Key'
    ], errors='ignore').groupby('Country/Region').sum()

    df = pd.merge(countries[['name']], df, how='inner', left_on=['name'], right_on=['Country/Region'])
    df.index = countries[countries['name'].isin(df['name'])].index
    df = df.drop(columns=['name'])

    return df


def combine_time_series(confirmed, deaths):
    df = pd.DataFrame(
        [
            [date, country, num_confirmed, deaths[date][country]]
            for date in confirmed.columns
            for num_confirmed, country in zip(confirmed[date], confirmed.index)
        ],
        columns=['date', 'country', 'confirmed', 'deaths']
    )

    df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)

    return df


def read_data(update=False):
    if update:
        update_data()

    countries = pd.read_csv('data/countries.csv', index_col='country')

    confirmed_global = read_time_series(countries, 'data/time_series/confirmed.csv')
    deaths_global = read_time_series(countries, 'data/time_series/deaths.csv')
    data_global = combine_time_series(confirmed_global, deaths_global)

    confirmed_ru = read_time_series(countries, 'data/time_series/confirmed_ru.csv')
    deaths_ru = read_time_series(countries, 'data/time_series/deaths_ru.csv')
    data_ru = combine_time_series(confirmed_ru, deaths_ru)

    data = data_global.append(data_ru)
    data = data[data['confirmed'] > 0]
    data = data.sort_values(by=['country', 'date']).reset_index(drop=True)

    data = data[data['date'] <= data[data['country'] == 'RU-MOW']['date'].max()]

    return countries, data
