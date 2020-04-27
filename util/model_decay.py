import numpy as np
import pandas as pd


def predict_decay(data, num_days=90, decay=0.99):
    date_to = data['date'].max()
    date_from = date_to - pd.to_timedelta(3, unit='d')

    data_to = data[data['date'] == date_to][['country', 'confirmed', 'deaths']]
    data_from = data[data['date'] == date_from][['country', 'confirmed', 'deaths']]

    data_avg = pd.merge(data_to, data_from, on='country')
    data_avg['confirmed'] = (data_avg['confirmed_x'] - data_avg['confirmed_y']) / 3
    data_avg['deaths'] = ((data_avg['deaths_x'] - data_avg['deaths_y']) / 3)

    data_avg = data_avg.set_index('country')[['confirmed', 'deaths']]
    data_to = data_to.set_index('country')

    preds = pd.DataFrame()
    for i in range(1, num_days + 1):
        date_data = (data_to + i * data_avg - decay * data_avg * np.sum([x for x in range(i)]) / num_days).copy()
        date_data['date'] = date_to + pd.to_timedelta(i, unit='d')

        preds = preds.append(date_data.reset_index()[['date', 'country', 'confirmed', 'deaths']], ignore_index=True)

    return preds.rename(columns={
        'confirmed': 'prediction_confirmed',
        'deaths': 'prediction_deaths',
    })
