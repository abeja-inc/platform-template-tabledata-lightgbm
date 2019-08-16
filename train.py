# coding: utf-8
# Template: 

import os
import gc
import random
from pathlib import Path
from math import modf

import pandas as pd
import lightgbm as lgb

from abeja.datalake import Client as DatalakeClient



# Configurable, but you don't need to set this. Default is "~/.abeja/.cache" in ABEJA Platform.
ABEJA_STORAGE_DIR_PATH = os.getenv('ABEJA_STORAGE_DIR_PATH')

DATALAKE_CHANNEL_ID = os.getenv('DATALAKE_CHANNEL_ID')
DATALAKE_TRAIN_ID = os.getenv('DATALAKE_TRAIN_ID')
DATALAKE_TEST_ID = os.getenv('DATALAKE_TEST_ID')

# params for load
USECOLS = os.getenv('USECOLS')
TARGET = os.getenv('TARGET') # required

if TARGET is None:
    raise Exception(f'TARGET is required')

# params for lgb
PARAMS = os.getenv('PARAMS')
params = {}
for kv in PARAMS.split(','):
    k, v = kv.split('=')
    
    try:
        if v in ['True', 'False']:
            params[k] = bool(v)
        elif v == 'None':
            params[k] = None
        else:
            # int or float
            decimal, integer = modf(float(v))
            if decimal == 0:
                params[k] = int(v)
            else:
                params[k] = float(v)
    except:
        params[k] = v




# =============================================================================
# 
# =============================================================================


def handler(context):
    print('Start train handler.')
    
    # load train
    datalake_client = DatalakeClient()
    channel = datalake_client.get_channel(DATALAKE_CHANNEL_ID)
    datalake_file = channel.get_file(DATALAKE_TRAIN_ID)
    datalake_file.get_content(cache=True)
    
    csvfile = Path(ABEJA_STORAGE_DIR_PATH, DATALAKE_CHANNEL_ID, DATALAKE_TRAIN_ID)
    if USECOLS is None:
        train = pd.read_csv(csvfile)
    else:
        usecols = USECOLS.split(',')
        train = pd.read_csv(csvfile, usecols=usecols+[TARGET])
    
    y_train = train[TARGET].values
    cols_drop = [c for c in train.columns if train[c].dtype == 'O'] + [TARGET]
    train.drop(cols_drop, axis=1, inplace=True)
    X_train = train
    cols_train = X_train.columns.tolist()
    del train
    
    dtrain = lgb.Dataset(X_train, y_train)
    
    model = lgb.train(params, train_set=dtrain, #num_boost_round=num_boost_round, 
                      valid_sets=dtrain, valid_names=None, fobj=None, feval=None, 
                      init_model=None, feature_name='auto', categorical_feature='auto', 
                      early_stopping_rounds=None, evals_result=None, 
                      verbose_eval=True, learning_rates=None, 
                      keep_training_booster=False, callbacks=None)
    
    del dtrain, X_train; gc.collect()
    
    # load test
    if DATALAKE_TEST_ID is not None:
        datalake_client = DatalakeClient()
        channel = datalake_client.get_channel(DATALAKE_CHANNEL_ID)
        datalake_file = channel.get_file(DATALAKE_TEST_ID)
        datalake_file.get_content(cache=True)
        
        csvfile = Path(ABEJA_STORAGE_DIR_PATH, DATALAKE_CHANNEL_ID, DATALAKE_TEST_ID)
        X_test = pd.read_csv(csvfile, usecols=cols_train)[cols_train]
        pred = model.predict(X_test)
        
        print(pred)
    

if __name__ == '__main__':
    handler()
