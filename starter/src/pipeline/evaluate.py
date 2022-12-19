import sys
import logging

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from pipeline.helpers.model import get_inference_pipeline, compute_model_metrics
from pipeline.helpers.model import split_features_response

logging.basicConfig(stream=sys.stdout, 
                    level=logging.INFO)

def go(model_artifact, df_train, df_test, params):

    logging.info("::evaluating model performance")
    for (df_data, dataset_name) in zip([df_train, df_test], 
                                       ['trainset','testset']):
        df_data = df_data.copy()

        df_X, df_y = split_features_response(df_data, 'high_salary')

        y_pred = model_artifact.predict(df_X)
        precision, recall, fbeta = compute_model_metrics(df_y.values, y_pred)

        logging.info("::  (dataset=%s): precision %0.2f, recall: %0.2f, fbeta: %0.2f"%
                     (dataset_name, precision, recall, fbeta))

    logging.info("::evaluating model performance on slices")

    ## evaluate on data slices
    occupation_list=df_test['occupation'].unique().tolist()

    for occupation in occupation_list:
        df_X, df_y = split_features_response(df_data.loc[df_data['occupation']==occupation], 'high_salary')

        y_pred = model_artifact.predict(df_X)
        precision, recall, fbeta = compute_model_metrics(df_y.values, y_pred)

        logging.info("::  (occupation == %s): precision %0.2f, recall: %0.2f, fbeta: %0.2f"%
                     (occupation, precision, recall, fbeta))
