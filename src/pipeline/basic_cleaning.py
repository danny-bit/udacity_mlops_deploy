import sys
import logging
import numpy as np
import pandas as pd

logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO)


def go(input_artifact):

    logging.info("::cleaning data...")
    # read with whitespaces around delimiter
    df_data = pd.read_csv(input_artifact,
                          sep='\s*,\s+',
                          engine='python')

    logging.info("::number of samples: %d" % (len(df_data)))
    # COLUMN NAMES
    column_names = [col_name.replace('-', '_') for col_name in df_data.columns]
    df_data.columns = column_names

    # Survey of Income and Program Participation (SIPP) includes person weights
    # that estimate the number of people in the target population
    # that each person represents.
    df_data = df_data.rename(columns={'fnlgt': 'final_weight'})

    # NAN values
    df_data['workclass'] = df_data['workclass'].replace('?', np.nan)
    df_data['occupation'] = df_data['occupation'].replace('?', np.nan)
    df_data = df_data.dropna()

    # OTHER
    # drop education column - redundandant with education num_column
    df_data = df_data.drop('education', axis=1)

    df_data = df_data[~df_data.duplicated()]

    # encode response as binary
    df_data['high_salary'] = df_data['salary'].map({'>50K': 1,
                                                    '<=50K': 0})
    df_data = df_data.drop('salary', axis=1)
    print(df_data[df_data['high_salary'] == 1].sample(3))

    return df_data
