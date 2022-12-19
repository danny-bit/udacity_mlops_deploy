import sys
import logging

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from pipeline.helpers.model import get_inference_pipeline, compute_model_metrics, split_features_response

logging.basicConfig(stream=sys.stdout, 
                    level=logging.INFO)

def go(input_artifact, params):
    logging.info("::model training...")

    df_trainval = input_artifact.copy()

    cat_features = [
        "workclass",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]
    num_features = [
        'age',
        'education_num',
        #'capital_gain',
        #'capital_loss',
        'hours_per_week'
    ]

    features = {}
    features['categorical'] = cat_features
    features['numerical'] = num_features

    #print(df_trainval.describe())
    #for feature in features['categorical']:
    #    print(feature)
    #    print(df_trainval[feature].unique())

    logging.info(f"::training samples {len(df_trainval)}...")

    inference_pipeline = get_inference_pipeline(features=features)

    df_X, df_y = split_features_response(df_trainval, 'high_salary')
    inference_pipeline.fit(df_X, df_y)

    return inference_pipeline