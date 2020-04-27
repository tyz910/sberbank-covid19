import numpy as np

COUNTRY_FEATURES = ["population", "area", "density", "lat", "long", "is_region"]
DAYS_SINCE_CONFIRMED = [1, 10, 50, 100, 500, 1000, 5000, 10000]


def add_features(data, countries):
    add_prev_values(data)
    add_days_since_confirmed(data, DAYS_SINCE_CONFIRMED)
    add_country_features(data, countries, COUNTRY_FEATURES)
    data.fillna(0, inplace=True)


def add_prev_values(data, num_days=40):
    for i in range(1, num_days + 1):
        for col in ['confirmed', 'deaths']:
            data[f"{col}_prev{i}"] = data.groupby("country")[col].shift(i)

    data.fillna(0, inplace=True)


def add_days_since_confirmed(data, days):
    for i in days:
        data[f'days_since_{i}_confirmed'] = 0
        data_confirmed = data[data['confirmed'] >= i]
        for c in data_confirmed['country'].unique():
            idx = data_confirmed[data_confirmed['country'] == c].index
            data.loc[idx, f'days_since_{i}_confirmed'] = np.arange(len(idx))


def add_country_features(data, countries, features):
    data[features] = data.join(countries, how='left', on='country')[features]


def get_Xy(data, gap):
    days_since_features = [f'days_since_{i}_confirmed' for i in DAYS_SINCE_CONFIRMED]

    X = data[COUNTRY_FEATURES + days_since_features].copy()
    X['perc_confirmed'] = data[f'confirmed_prev{gap}'] / data['population']

    for c in ['confirmed', 'deaths']:
        X[f'{c}_prev'] = data[f'{c}_prev{gap}']

        for i in range(3):
            X[f'diff_{i + 1}_{c}'] = data[f'{c}_prev{gap + i}'] - data[f'{c}_prev{gap + i + 1}']
            X[f'change_{i + 1}_{c}'] = (data[f'{c}_prev{gap + i}'] + 1) / (data[f'{c}_prev{gap + i + 1}'] + 1)

        for i in [1, 2]:
            X[f'diff_change_{i}_{c}'] = (X[f'diff_{i}_{c}'] + 1) / (X[f'diff_{i + 1}_{c}'] + 1)
        X[f'diff_change_12_{c}'] = (X[f'diff_change_1_{c}'] + X[f'diff_change_2_{c}']) / 2

        X[f'diff_123_{c}'] = (data[f'{c}_prev{gap}'] - data[f'{c}_prev{gap + 3}']) / 3
        X[f'change_1_3_{c}'] = (data[f'{c}_prev{gap}'] + 1) / (data[f'{c}_prev{gap + 3}'] + 1)
        X[f'change_1_7_{c}'] = (data[f'{c}_prev{gap}'] + 1) / (data[f'{c}_prev{gap + 7}'] + 1)

    for f in days_since_features:
        X.loc[X[f] < gap, f] = 0

    y = data[['confirmed', 'deaths']].copy()
    y -= data[[f'confirmed_prev{gap}', f'deaths_prev{gap}']].values
    y.loc[y['confirmed'] < 0, 'confirmed'] = 0
    y.loc[y['deaths'] < 0, 'deaths'] = 0

    return X, np.log10(y + 1.0)
