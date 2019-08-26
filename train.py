# coding: utf-8
# Template: 

import os
import gc
import json
from pathlib import Path
from math import modf

import pandas as pd
import numpy as np
import lightgbm as lgb

from abeja.datalake import Client as DatalakeClient



# Configurable, but you don't need to set this. Default is "~/.abeja/.cache" in ABEJA Platform.
ABEJA_STORAGE_DIR_PATH = os.getenv('ABEJA_STORAGE_DIR_PATH')
ABEJA_TRAINING_RESULT_DIR = os.getenv('ABEJA_TRAINING_RESULT_DIR')

DATALAKE_CHANNEL_ID = os.getenv('DATALAKE_CHANNEL_ID')
DATALAKE_TRAIN_ITEM_ID = os.getenv('DATALAKE_TRAIN_ITEM_ID')
DATALAKE_TEST_ITEM_ID = os.getenv('DATALAKE_TEST_ITEM_ID')

# params for load
INPUT_FIELDS = os.getenv('INPUT_FIELDS')
TARGET_FIELD = os.getenv('TARGET_FIELD') # required

if TARGET_FIELD is None:
    raise Exception(f'TARGET is required')

# params for lgb
PARAMS = os.getenv('PARAMS')
params = {}
if PARAMS is None:
    pass
elif len(PARAMS) == 0:
    pass
else:
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

NFOLD = int(os.getenv('NFOLD', '5'))

EARLY_STOPPING_ROUNDS = os.getenv('EARLY_STOPPING_ROUNDS')
if EARLY_STOPPING_ROUNDS is not None:
    EARLY_STOPPING_ROUNDS = int(EARLY_STOPPING_ROUNDS)

VERBOSE_EVAL = os.getenv('VERBOSE_EVAL')
if VERBOSE_EVAL is not None:
    VERBOSE_EVAL = int(VERBOSE_EVAL)

STRATIFIED  = os.getenv('STRATIFIED')
if STRATIFIED is not None:
    STRATIFIED = bool(STRATIFIED)
else:
    STRATIFIED = True

# =============================================================================
# 
# =============================================================================
class ModelExtractionCallback(object):
    """
    original author : momijiame
    ref : https://blog.amedama.jp/entry/lightgbm-cv-model
    description : Class for callback to extract trained models from lightgbm.cv(). 
    note: This class depends on private class '_CVBooster', so there are some future risks. 
    
    usage:
        extraction_cb = ModelExtractionCallback()
        callbacks = [extraction_cb,]
    
        lgb.cv(params, dtrain, nfold=5, 
               num_boost_round=9999,
               early_stopping_rounds=EARLY_STOPPING_ROUNDS,
               verbose_eval=verbose_eval,
               callbacks=callbacks,
               seed=0)
        
        models = extraction_cb.raw_boosters
    
    """

    def __init__(self):
        self._model = None

    def __call__(self, env):
        # _CVBooster の参照を保持する
        self._model = env.model

    def _assert_called_cb(self):
        if self._model is None:
            # コールバックが呼ばれていないときは例外にする
            raise RuntimeError('callback has not called yet')

    @property
    def boosters_proxy(self):
        self._assert_called_cb()
        # Booster へのプロキシオブジェクトを返す
        return self._model

    @property
    def raw_boosters(self):
        self._assert_called_cb()
        # Booster のリストを返す
        return self._model.boosters

    @property
    def best_iteration(self):
        self._assert_called_cb()
        # Early stop したときの boosting round を返す
        return self._model.best_iteration

def handler(context):
    print('Start train handler.')
    
    # load train
    datalake_client = DatalakeClient()
    channel = datalake_client.get_channel(DATALAKE_CHANNEL_ID)
    datalake_file = channel.get_file(DATALAKE_TRAIN_ITEM_ID)
    datalake_file.get_content(cache=True)
    
    csvfile = Path(ABEJA_STORAGE_DIR_PATH, DATALAKE_CHANNEL_ID, DATALAKE_TRAIN_ITEM_ID)
    if INPUT_FIELDS is None:
        train = pd.read_csv(csvfile)
    else:
        usecols = INPUT_FIELDS.split(',')
        train = pd.read_csv(csvfile, usecols=usecols+[TARGET_FIELD])
    
    y_train = train[TARGET_FIELD].values
    cols_drop = [c for c in train.columns if train[c].dtype == 'O'] + [TARGET_FIELD]
    train.drop(cols_drop, axis=1, inplace=True)
    X_train = train
    cols_train = X_train.columns.tolist()
    del train
    
    dtrain = lgb.Dataset(X_train, y_train)
    
    extraction_cb = ModelExtractionCallback()
    callbacks = [extraction_cb,]
    
    lgb.cv(params, dtrain, nfold=NFOLD,
           early_stopping_rounds=EARLY_STOPPING_ROUNDS,
           verbose_eval=VERBOSE_EVAL,
           stratified=STRATIFIED,
           callbacks=callbacks,
           seed=0)
    
    models = extraction_cb.raw_boosters
    for i,model in enumerate(models):
        model.save_model(os.path.join(ABEJA_TRAINING_RESULT_DIR, f'model_{i}.txt'))
    
    di = {
            'NFOLD': NFOLD,
            'EARLY_STOPPING_ROUNDS': EARLY_STOPPING_ROUNDS,
            'VERBOSE_EVAL': VERBOSE_EVAL,
            'STRATIFIED': STRATIFIED,
            'cols_train': cols_train
            
        }
    lgb_env = open(os.path.join(ABEJA_TRAINING_RESULT_DIR, 'lgb_env.json'), 'w')
    json.dump(di, lgb_env)
    lgb_env.close()
    
    del dtrain, X_train; gc.collect()
    
    # load test
    if DATALAKE_TEST_ITEM_ID is not None:
        datalake_client = DatalakeClient()
        channel = datalake_client.get_channel(DATALAKE_CHANNEL_ID)
        datalake_file = channel.get_file(DATALAKE_TEST_ITEM_ID)
        datalake_file.get_content(cache=True)
        
        csvfile = Path(ABEJA_STORAGE_DIR_PATH, DATALAKE_CHANNEL_ID, DATALAKE_TEST_ITEM_ID)
        X_test = pd.read_csv(csvfile, usecols=cols_train)[cols_train]
        
        pred = np.zeros(len(X_test))
        for model in models:
            pred += model.predict(X_test)
        pred /= len(models)
        
        print(pred)
    

if __name__ == '__main__':
    handler()
