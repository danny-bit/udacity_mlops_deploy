from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score


def get_inference_pipeline(features,
                           hyperparameters=None):

    model = RandomForestClassifier()

    categ_preproc = make_pipeline(OrdinalEncoder(handle_unknown="use_encoded_value",
                                                 unknown_value=1000))
    numeric_preproc = make_pipeline(StandardScaler())

    preprocessor = ColumnTransformer([
        ('categorical', categ_preproc, features['categorical']),
        ('numerical', numeric_preproc, features['numerical'])],
        remainder='drop')

    inference_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    return inference_pipe


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """

    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)

    return precision, recall, fbeta


def split_features_response(df_data, response_column):
    """split a dataframe into a features dataframe and the response column

    Args:
        df_data (pandas DataFrame): input dataframe
        response_column (str): name of the response column

    Returns:
        df_X: dataframe of the features
        df_y: dataframe of the response column
    """
    df_y = df_data[response_column]
    feature_columns = [col_name for col_name in df_data.columns
                       if not col_name == response_column]
    df_X = df_data.loc[:, feature_columns]
    return (df_X, df_y)
