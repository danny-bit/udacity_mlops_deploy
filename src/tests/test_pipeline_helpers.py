import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from src.pipeline.helpers.model import compute_model_metrics, get_inference_pipeline
from src.pipeline.helpers.model import split_features_response


def test_compute_model_metric():
    """
    test the compute_model_metric pipeline function of src.pipeline.helpers
    """
    n_samples = 3
    precision, recall, fbeta = compute_model_metrics(y=np.ones(n_samples,),
                                                     preds=np.ones(n_samples,))
    assert (precision == 1)
    assert (recall == 1)
    assert (fbeta == 1)


def test_get_inference_pipeline():
    """
    test the get_inference_pipeline function of src.pipeline.helpers
    """

    features = {}
    features['categorical'] = 'cat_test'
    features['numerical'] = 'num_test'
    pipeline = get_inference_pipeline(features)

    assert (isinstance(pipeline, Pipeline))
    assert 'cat_test' in pipeline.named_steps['preprocessor'].__repr__()
    assert 'num_test' in pipeline.named_steps['preprocessor'].__repr__()


def test_split_features_response():
    df_random = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)),
                             columns=list('ABCD'))
    df_X, df_y = split_features_response(df_random, 'D')

    assert not 'D' in df_X.columns
    assert (df_y.values == df_random['D']).all()
