import numpy as np
import pandas as pd
import lightgbm as lgb

from util.features import add_features, get_Xy


def predict_lgb(data, countries, model_params, predict_days, with_deaths=True):
    data_train, data_test = data_split_train_test(data, countries, predict_days)

    return train_and_predict_lgb(data_train, data_test, model_params, with_deaths)


def predict_lgb_iterate(data, countries, model_params, predict_days, iterate_days, with_deaths=True):
    preds = pd.DataFrame()

    while True:
        preds = preds.append(predict_lgb(data, countries, model_params, iterate_days, with_deaths), ignore_index=True)

        if len(preds['date'].unique()) >= predict_days:
            break

        data = data.append(preds.rename(columns={
            'prediction_confirmed': 'confirmed',
            'prediction_deaths': 'deaths',
        }), ignore_index=True)

    max_date = preds['date'].unique()[predict_days - 1]

    return preds[preds['date'] <= max_date].reset_index(drop=True)


def train_and_predict_lgb(data_train, data_test, model_params, with_deaths=True):
    preds = pd.DataFrame(columns=['date', 'country', 'prediction_confirmed', 'prediction_deaths'])

    for gap, date in enumerate(data_test['date'].unique(), 1):
        X_train, y_train = get_Xy(data_train, gap)
        X_test, _ = get_Xy(data_test[data_test['date'] == date], gap)

        model_confirmed, model_deaths = train_lgb_models(X_train, y_train, model_params, with_deaths)
        preds_confirmed = model_confirmed.predict(X_test)

        if with_deaths:
            preds_deaths = model_deaths.predict(X_test)
        else:
            preds_deaths = np.zeros(len(preds_confirmed))

        preds = preds.append(pd.DataFrame({
            'date': date,
            'country': data_test[data_test['date'] == date]['country'].values,
            'prediction_confirmed': X_test['confirmed_prev'] + 10.0 ** preds_confirmed - 1.0,
            'prediction_deaths': X_test['deaths_prev'] + 10.0 ** preds_deaths - 1.0,
        })).reset_index(drop=True)

    return preds


def train_lgb_models(X, y, model_params, with_deaths=True):
    lgb_params = model_params.copy()
    lgb_params['objective'] = 'regression'
    lgb_params['metric'] = 'rmse'
    del lgb_params['regions_weight']
    del lgb_params['num_boost_round']

    weight = (X['is_region'] + 1.0).copy()
    weight.loc[weight > 1.0] = model_params['regions_weight']

    train_confirmed = lgb.Dataset(X, label=y['confirmed'], weight=weight)
    model_confirmed = lgb.train(lgb_params, train_set=train_confirmed, num_boost_round=model_params['num_boost_round'])

    if with_deaths:
        train_deaths = lgb.Dataset(X, label=y['deaths'])
        model_deaths = lgb.train(lgb_params, train_set=train_deaths, num_boost_round=model_params['num_boost_round'])
    else:
        model_deaths = None

    return model_confirmed, model_deaths


def data_split_train_test(data, countries, predict_days):
    max_date = data['date'].max()

    data_test = pd.DataFrame([
        [max_date + pd.to_timedelta(d + 1, unit='d'), c, 0, 0]
        for c in data['country'].unique()
        for d in range(predict_days)
    ], columns=['date', 'country', 'confirmed', 'deaths'])

    data_full = data.append(data_test).sort_values(by=['country', 'date']).reset_index(drop=True).copy()
    add_features(data_full, countries)

    data_train = data_full[(data_full['date'] <= max_date) & (data_full['confirmed'] > 0)].reset_index(drop=True)
    data_test = data_full[data_full['date'] > max_date].reset_index(drop=True)

    return data_train, data_test
