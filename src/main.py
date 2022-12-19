"""
    execute the model pipeline
"""

import sys
import logging
import joblib
import pipeline.basic_cleaning
import pipeline.train
import pipeline.evaluate

from sklearn.model_selection import train_test_split


def go():
    config = {'input_artifact': 'data/census.csv',
              'test_size': 0.15,
              'random_state': 1,
              'path_model_artifact': 'model/census_model.pkl',
              'stratify': 'high_salary'}

    df_cleaned = pipeline.basic_cleaning.go(
        input_artifact=config['input_artifact'])

    logging.info("::splitting data into trainval and test")
    df_trainval, df_test = train_test_split(df_cleaned,
                                            test_size=config['test_size'],
                                            random_state=config['random_state'],
                                            stratify=df_cleaned[config['stratify']])

    model = pipeline.train.go(input_artifact=df_trainval, params=[])

    pipeline.evaluate.go(model_artifact=model,
                         df_train=df_trainval,
                         df_test=df_test,
                         params=[])

    joblib.dump(model, config['path_model_artifact'])


if __name__ == '__main__':
    go()
