import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from src.pipeline.helpers.model import compute_model_metrics, get_inference_pipeline 

def test_compute_model_metric():
    """
    test the compute_model_metric pipeline function of src.pipeline.helpers
    """
    n_samples = 3
    precision, recall, fbeta = compute_model_metrics(y = np.ones(n_samples,), 
                                                     preds = np.ones(n_samples,))
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

# #def test_pop_response():
#     """
#     test the pop_response_function of src.pipeline.helpers
#     """

#     n_data = 3
#     df_art = pd.DataFrame()
#     df_art['feature'] = np.ones(n_data,)
#     df_art['response'] = np.ones(n_data,)

#     df_X, df_y = pop_response(df_data = df_art, 
#                               response_column='response')

#     assert (df_X.columns == 'feature')
#     assert (df_y.name == 'response')