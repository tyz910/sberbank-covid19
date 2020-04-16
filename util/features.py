import numpy as np


def add_prev_values(df, days):
    df = df.copy()

    for i in range(1, days + 1):
        for col in ['confirmed', 'deaths']:
            df[f"{col}_prev{i}"] = df.groupby("country")[col].shift(i)

    return df.fillna(0)


def add_days_since_start(df):
    df = df.copy()

    df['days_since_start'] = 0
    for c in df['country'].unique():
        df.loc[df['country'] == c, 'days_since_start'] = np.arange(len(df[df['country'] == c]))

    return df

