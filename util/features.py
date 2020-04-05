import numpy as np


def add_prev_values(df, col, days):
    for i in range(1, days + 1):
        df["{}_prev_{}".format(col, i)] = df.groupby("country")[col].shift(i)
    df.fillna(0, inplace=True)


def add_days_since_start(df):
    df['days_since_start'] = 0
    for c in df['country'].unique():
        df.loc[df['country'] == c, 'days_since_start'] = np.arange(len(df[df['country'] == c]))
