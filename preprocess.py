from __future__ import division

import pandas as pd
from sklearn import preprocessing


def dummify(df):
    # from data frame creates dummy variables; creates single column for binary category

    le = preprocessing.LabelEncoder()

    # make binary categories into 0/1 ints
    for col in df:
        if len(df[col].unique()) == 2 and df[col].dtype == 'object':
            le.fit(df[col])
            df[col] = le.transform(df[col])

    cat_df = df.select_dtypes(include=['object'])
    num_df = df.select_dtypes(exclude=['object'])

    cat_df = pd.get_dummies(cat_df)

    return pd.concat([num_df, cat_df], axis = 1)


def factorize(df):
    # from data frame creates integer factors; creates single column for binary category

    le = preprocessing.LabelEncoder()

    for col in df:
        if df[col].dtype == 'object':
            le.fit(df[col])
            df[col] = le.transform(df[col])
    return df