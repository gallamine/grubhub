# -*- coding: utf-8 -*-

"""Main module."""
import daiquiri
import logging
import pandas as pd
import json
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from typing import Union


daiquiri.setup(level=logging.INFO)


def get_data_source(data_set:str='default') -> (str,list):
    """
    Return the resource location for the data we'd like to grab.
    Data is retrieved from a data definition file in `data/training_data.json`.

    Returns: formatted URL of the data we'd like to grab, list of the required colunn names

    """

    with open(os.path.join('data','training_data.json')) as f:
        data_definitions = json.load(f)

    training_data_defnition = data_definitions.get(data_set,'default')

    try:
        training_data_location = training_data_defnition['url'].format(training_data_defnition['doc_id'])
        return training_data_location, training_data_defnition['columns']

    except KeyError:
        logging.exception(f"Missing training data definition (`doc_id` and `url`) for data set {data_set}")

    except:
        raise


def validate_data(data_df:pd.DataFrame, req_column_names:list) -> bool:
    """
    TODO
    Args:
        data_df:

    Returns: True if the imported data is valid

    """

    for column in req_column_names:
        if column not in data_df.columns:
            logging.error(f"Rquired column {column} is missing from the data. Was given {list(data_df.columns)}")
            raise KeyError

    # TODO Do other validation here

    return True


def load_data(data_source:str='default', use_cache:bool=False) -> pd.DataFrame:
    """

    Args:
        url:
        use_cache:

    Returns:

    """
    if use_cache:
        raise NotImplementedError

    else:

        data_url, req_column_names = get_data_source('default')
        df = pd.read_csv(data_url, parse_dates=['date'])

        if validate_data(df, req_column_names):
            return df
        else:
            # TODO We need a custom exception for invalid model data
            raise Exception('Training data is invalid')


def build_features(data_df: pd.DataFrame, region:int=1) -> pd.DataFrame:
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

    except Exception as e:
        logging.exception(f"There was an error building featues for {data-df}")


def train_model(df_train, predictors:list, n_trees:int=50):
    """

    Args:
        predictors:
        df_train:

    Returns:

    """

    try:

        rf = RandomForestRegressor(n_trees, min_samples_leaf=1)

        rf.fit(df_train[predictors], df_train.order_count)

        df_train.loc[:, 'predicted'] = rf.predict(df_train[predictors])
        df_train.loc[:, 'residual'] = df_train.order_count - df_train.predicted

        return rf, df_train

    except Exception as e:
        logging.exception("There was an error training the model.")
        raise


def evaluate_model(model:RandomForestRegressor, df_train) -> RandomForestRegressor:
    """

    Args:
        mode:
        df_train:

    Returns:

    """

    rmse_1 = np.sqrt(df_train.residual.pow(2).mean())
    setattr(model, '_model_info', {'performance':{'rmse':rmse_1}})
    return model


def persist_model(model:RandomForestRegressor, location:str='tmp/trained_model.pickle') -> str:
    """

    Args:
        model:

    Returns:

    """
    import pickle

    # TODO Take the hash of the model and use as persistent name


    with open(os.path.join(location),'wb') as f:
        pickle.dump(model, f)



def train(region:str) -> str:
    """


    Args:
        region:

    Returns:

    """
    logger = daiquiri.getLogger(__name__)
    logger.info(f"Training the model for region {region}!!")

    df = load_data(data_source='default', use_cache=False)
    df = build_features(df, region=region)
    model, df = train_model(df, predictors = ['weekday', 'month', 'event', 'time'])
    model = evaluate_model(model, df)
    persist_model(model)

