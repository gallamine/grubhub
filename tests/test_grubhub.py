from grubhub import grubhub
import pytest
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import os


def test_get_data_source():

    data_url, req_columns = grubhub.get_data_source('default')
    assert data_url == 'https://drive.google.com/uc?id=1vKJ0GIKw18td9iHpKPa5N14fFsrBaYJY&export=dowload'
    assert req_columns == ["date", "event", "order_count", "region_id"]

    with pytest.raises(KeyError):
        grubhub.get_data_source('bad_source')

def test_validate_data():

    test_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    assert grubhub.validate_data(test_df, req_column_names=['a', 'b'])

    with pytest.raises(KeyError):
        assert grubhub.validate_data(test_df, req_column_names=['not', 'there'])


def test_load_data(monkeypatch):

    def mock_get_data_source(region, use_cache=False):
        return 'tests/test_data/test_csv_dataframe.csv', ['date', 'count']

    monkeypatch.setattr(grubhub, 'get_data_source', mock_get_data_source)

    with pytest.raises(NotImplementedError):
        assert grubhub.load_data('cached_url', use_cache=True)

    assert isinstance(grubhub.load_data(), pd.DataFrame)


def test_build_features():
    """
    Should create data test fixtures for tests like these.
    Returns:

    """

    test_df = pd.DataFrame({'date':
                            pd.to_datetime(
                                    pd.Series(['1/1/2018',
                                               '1/2/2018',
                                               '1/3/2018',
                                               '1/4/2018',
                                               '1/5/2018',
                                               '1/6/2018'])),
                            'region_id':
                                [1, 1, 1, 2, 2, 2]})

    features_df = grubhub.build_features(test_df, region=1)

    assert 'month' in features_df.columns
    assert 'weekday' in features_df.columns
    assert 'time' in features_df.columns
    assert len(features_df) == 3


def test_train_model():

    test_df = pd.DataFrame({'order_count': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                            'feature_a': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
                            'feature_b': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]})

    model, trained_df = grubhub.train_model(test_df,
                                            predictors=['feature_a', 'feature_b'],
                                            n_trees=5)

    assert 'predicted' in trained_df.columns
    assert 'residual' in trained_df.columns
    assert isinstance(model, RandomForestRegressor)


def test_persist_model():

    test_file_loc = '/tmp/test_clf.pickle'
    grubhub.persist_model(RandomForestRegressor(), location=test_file_loc)

    assert os.path.isfile(test_file_loc)

    os.remove(test_file_loc)
