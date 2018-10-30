# -*- coding: utf-8 -*-

"""Main module."""
import daiquiri
import logging
import pandas as pd
import json
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import time

daiquiri.setup(level=logging.INFO)


def get_data_source(data_set: str = 'default') -> (str, list, list):
    """
    Return the resource location for the data we'd like to grab.
    Data is retrieved from a data definition file in `data/training_data.json`.

    Args:
        data_set: dataset defined in `data/training_data.json`. Default is `default`

    Returns: formatted URL of the data we'd like to grab, list of the required colunn names

    """
    base_dir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(base_dir, 'data', 'training_data.json')) as f:
        data_definitions = json.load(f)

    try:
        training_data_definition = data_definitions.get(data_set, {})
        training_data_location = training_data_definition['url'].format(training_data_definition['doc_id'])
        return training_data_location, training_data_definition['columns'], training_data_definition['dtypes']

    except KeyError:
        logging.exception(f"Missing training data definition (`doc_id` and `url`) for data set {data_set}")
        raise

    except Exception:
        raise


def validate_data(data_df: pd.DataFrame, req_column_names: list, req_column_types: list) -> bool:
    """
    This function validates the input data. The amount of validation is depended on conversations
    with the data scientist building the model. Currently we just check that the required columns are present.

    Args:
        data_df: raw data dataframe
        req_column_names: required column names
        req_column_types: expected numpy dtypes of each column

    Returns: True if the imported data is valid

    """

    if len(req_column_names) != len(req_column_types):
        raise Exception("column names and column types must be the same length.")

    for column in req_column_names:
        if column not in data_df.columns:
            logging.error(f"Required column {column} is missing from the data. Was given {list(data_df.columns)}")
            raise KeyError

    for column_type, column_name in zip(req_column_types, req_column_names):
        if np.dtype(column_type) != data_df.dtypes[column_name]:
            raise TypeError(f"column {column_name} is of an invalid type, {data_df.dtypes[column_name]}")

    # TODO Do other validation here

    return True


def load_data(data_source: str = 'default', use_cache: bool = False) -> pd.DataFrame:
    """

    Load the data from a url defined by the `data_source` in `data/training_data.json`.
    Make sure the data passes some basic validation steps before proceeding.

    Args:
        use_cache: Read the data from cache?
        data_source: Name of source data defined in data/training_data.json

    Returns: DataFrame of the valid loaded data.

    """
    if use_cache:
        raise NotImplementedError

    else:

        data_url, req_column_names, req_column_types = get_data_source(data_source)
        df = pd.read_csv(data_url, parse_dates=['date'])

        if validate_data(df, req_column_names, req_column_types):
            return df
        else:
            # TODO We need a custom exception for invalid model data
            raise Exception('Training data is invalid')


def build_features(data_df: pd.DataFrame, region: int = 1) -> pd.DataFrame:
    """

    Get a dataframe raw input, build training features from this dataset and return the full dataframe with features.
    Build features for the given region.

    TODO: Make better idomatic pandas
    Args:
        data_df: raw data dataframe
        region: the region indicator for which data to build features for

    Returns: dataframe for the specific region containing the features

    """

    try:
        data_df.loc[:, 'month'] = data_df.date.dt.month
        data_df.loc[:, 'weekday'] = data_df.date.dt.weekday

        df = data_df.sort_values('date')

        df.loc[df.region_id == region, 'time'] = range(0, len(df.loc[df.region_id == region]))

        return df.loc[df.region_id == region]

    except Exception:
        logging.exception(f"There was an error building featues for {data_df}")
        raise


def train_model(df_train, predictors: list, n_trees: int = 50) -> (RandomForestRegressor, pd.DataFrame):
    """
    Train a RandomForestRegression using the colunns of `df_train` listed in `predictors`.
    Args:
        predictors: list of column names to use in input to RF
        df_train: training matrix
        n_trees: number of trees to use in the regression

    Returns:
        trained model and dataframe with the predicted and residual colunn added

    """

    try:

        rf = RandomForestRegressor(n_trees, min_samples_leaf=1)

        rf.fit(df_train[predictors], df_train.order_count)

        df_train.loc[:, 'predicted'] = rf.predict(df_train[predictors])
        df_train.loc[:, 'residual'] = df_train.order_count - df_train.predicted

        return rf, df_train

    except Exception:
        logging.exception("There was an error training the model.")
        raise


def evaluate_model(model: RandomForestRegressor, df_train) -> (RandomForestRegressor, float):
    """
    Compute the RMSE from the `residual` column of the input data_frame and store as part of the model so we have a
     record of the particular models performance.

    Args:
        model: Model object that we'll set a `_model_info` attribute on containing a dictionary describing the
         performance.
        df_train: the training data with a `residual` column that we use to calculate the RMSE of the classifier

    Returns: modified model object, rmse value

    """

    rmse_1 = np.sqrt(df_train.residual.pow(2).mean())
    setattr(model, '_model_info', {'performance': {'rmse': rmse_1}})
    return model, rmse_1


def persist_model(model: RandomForestRegressor, location: str = '/tmp/trained_model.pickle') -> str:
    """
    Take a RF model object and pickle to disk. Make sure we can unpickle it from the same location.

    TODO: 1) capture model hash for immutable storage 2) give optional GZIP compression 3) use different serializer
     that is safer

    Args:
        model: the RandomForest model object
        location: location where we'll serialize on disk

    Returns: location of the model on disk

    """
    import pickle

    # TODO Take the hash of the model and use as persistent name

    with open(os.path.join(location), 'wb') as f:
        pickle.dump(model, f)

    with open(os.path.join(location), 'rb') as f:
        try:
            pickle.load(f)
            return location
        except Exception:
            logging.exception("There was a problem serializing the model to disk")
            raise


def train(region: str) -> str:
    """

    Do the full training pipeline by downloading data, validating, building features and training a RandomForest
     regression, finally persist the model to disk.

    Args:
        region: region integer (1,2 are current options)

    Returns: location of serialized model

    """

    logger = daiquiri.getLogger(__name__)
    logger.info(f"Training the model for region {region}!")

    start_time = time.time()
    df = load_data(data_source='default', use_cache=False)
    df = build_features(df, region=region)
    model, df = train_model(df, predictors=['weekday', 'month', 'event', 'time'])
    model, rmse = evaluate_model(model, df)
    train_time = time.time() - start_time

    logging.info(f"Trained model for region {region} in {train_time} seconds with RMSE of {rmse}")
    model_loc = persist_model(model)

    return model_loc
