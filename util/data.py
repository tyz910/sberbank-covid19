import pandas as pd


def convert_countries(original_countries_path, confirmed_path, save_path):
    countries = pd.read_csv(original_countries_path, na_filter=False, index_col='iso_alpha2')
    countries['name'] = countries['ccse_name']
    countries = countries.drop(columns=['iso_alpha3', 'iso_numeric', 'official_name', 'ccse_name'])
    countries.index.name = 'country'

    df = pd.read_csv(confirmed_path, na_filter=False)
    df = df[['Country/Region', 'Lat', 'Long']]
    df = df.groupby('Country/Region').mean()

    df = pd.merge(countries, df, how='left', left_on=['name'], right_on=['Country/Region']).set_index(countries.index)
    df.to_csv(save_path)


def clean_time_series(df, countries):
    df = df.drop(columns=['Lat', 'Long', 'Province/State'])
    df = df.groupby('Country/Region').sum()
    df = pd.merge(countries[['name']], df, how='left', left_on=['name'], right_on=['Country/Region']).set_index(countries.index)
    df = df.drop(columns=['name'])

    return df


def read_data(countries_path, confirmed_path, deaths_path):
    countries = pd.read_csv(countries_path, index_col='country', keep_default_na=False, na_values=[''])
    confirmed = clean_time_series(pd.read_csv(confirmed_path, na_filter=False), countries)
    deaths = clean_time_series(pd.read_csv(deaths_path, na_filter=False), countries)

    data = [[date, country, num_confirmed, deaths[date][country]] for date in confirmed.columns for
            num_confirmed, country in zip(confirmed[date], confirmed.index)]

    df = pd.DataFrame(data, columns=['date', 'country', 'confirmed', 'deaths'])
    df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
    df = df[df['confirmed'] > 0]
    df = df.reset_index().drop(columns=['index'])

    return countries, df


def days_idx(df, days, after=True):
    date_from = df['date'].max() - pd.to_timedelta(days, unit='d')

    if after:
        return df[df['date'] > date_from].index
    else:
        return df[df['date'] <= date_from].index
