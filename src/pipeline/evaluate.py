import sys
import logging

from pipeline.helpers.model import compute_model_metrics
from pipeline.helpers.model import split_features_response

logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO)


def eval_on_feature_slices(model, df_data, feature_column):
    """evaluate the model metrics on feature slices

    Args:
        model (sklearn model): model that is used for inference
        df_data (pandas Dataframe): data for metric evaluation
        feature_column (str): name of the feature column
    """
    feature_value_list = df_data[feature_column].unique().tolist()

    with open('model/slice_output.txt', 'w+') as f:
        f.write('')

    for feature_value in feature_value_list:
        df_slice = df_data.loc[df_data[feature_column] == feature_value]
        df_X, df_y = split_features_response(df_slice, 'high_salary')

        y_pred = model.predict(df_X)
        precision, recall, fbeta = compute_model_metrics(df_y.values, y_pred)

        output_str = "::  (%s == %s): precision %0.2f, recall: %0.2f, fbeta: %0.2f" % \
                     (feature_column, feature_column, precision, recall, fbeta)

        with open('model/slice_output.txt', 'a') as f:
            f.write(output_str + '\r')
        logging.info(output_str)


def go(model_artifact, df_train, df_test, params):

    logging.info("::evaluating model performance")
    for (df_data, dataset_name) in zip([df_train, df_test],
                                       ['trainset', 'testset']):
        df_data = df_data.copy()

        df_X, df_y = split_features_response(df_data, 'high_salary')

        y_pred = model_artifact.predict(df_X)
        precision, recall, fbeta = compute_model_metrics(df_y.values, y_pred)

        logging.info("::  (dataset=%s): precision %0.2f, recall: %0.2f, fbeta: %0.2f" %
                     (dataset_name, precision, recall, fbeta))

    logging.info("::evaluating model performance on test slices")
    eval_on_feature_slices(model_artifact, df_data, 'occupation')
