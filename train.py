# coding: utf-8
# Template: 

import os
import gc
import json
from pathlib import Path

import pandas as pd
import numpy as np
import lightgbm as lgb
from abeja.datalake import Client as DatalakeClient
from tensorboardX import SummaryWriter

from callbacks import Statistics, TensorBoardCallback
from parameters import Parameters


ABEJA_STORAGE_DIR_PATH = Parameters.ABEJA_STORAGE_DIR_PATH
ABEJA_TRAINING_RESULT_DIR = Parameters.ABEJA_TRAINING_RESULT_DIR
DATALAKE_CHANNEL_ID = Parameters.DATALAKE_CHANNEL_ID
DATALAKE_TRAIN_FILE_ID = Parameters.DATALAKE_TRAIN_FILE_ID
DATALAKE_TEST_FILE_ID = Parameters.DATALAKE_TEST_FILE_ID
INPUT_FIELDS = Parameters.INPUT_FIELDS
LABEL_FIELD = Parameters.LABEL_FIELD
PARAMS = Parameters.as_params()

statistics = Statistics(Parameters.NUM_ITERATIONS)

log_path = os.path.join(ABEJA_TRAINING_RESULT_DIR, 'logs')
writer = SummaryWriter(log_dir=log_path)


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
    print(f'start training with parameters : {Parameters.as_dict()}, context : {context}')
    
    # load train
    datalake_client = DatalakeClient()
    channel = datalake_client.get_channel(DATALAKE_CHANNEL_ID)
    datalake_file = channel.get_file(DATALAKE_TRAIN_FILE_ID)
    datalake_file.get_content(cache=True)
    
    csvfile = Path(ABEJA_STORAGE_DIR_PATH, DATALAKE_CHANNEL_ID, DATALAKE_TRAIN_FILE_ID)
    if INPUT_FIELDS:
        train = pd.read_csv(csvfile, usecols=INPUT_FIELDS+[LABEL_FIELD])
    else:
        train = pd.read_csv(csvfile)

    y_train = train[LABEL_FIELD].values
    cols_drop = [c for c in train.columns if train[c].dtype == 'O'] + [LABEL_FIELD]
    train.drop(cols_drop, axis=1, inplace=True)
    X_train = train
    cols_train = X_train.columns.tolist()
    del train
    
    dtrain = lgb.Dataset(X_train, y_train)
    
    extraction_cb = ModelExtractionCallback()
    tensorboard_cb = TensorBoardCallback(statistics, writer)
    callbacks = [extraction_cb, tensorboard_cb,]
    
    lgb.cv(PARAMS, dtrain, nfold=Parameters.NFOLD,
           early_stopping_rounds=Parameters.EARLY_STOPPING_ROUNDS,
           verbose_eval=Parameters.VERBOSE_EVAL,
           stratified=Parameters.STRATIFIED,
           callbacks=callbacks,
           metrics=Parameters.METRIC,
           seed=Parameters.SEED)
    
    models = extraction_cb.raw_boosters
    for i,model in enumerate(models):
        model.save_model(os.path.join(ABEJA_TRAINING_RESULT_DIR, f'model_{i}.txt'))
    
    di = {
        **(Parameters.as_dict()),
        'cols_train': cols_train
    }
    lgb_env = open(os.path.join(ABEJA_TRAINING_RESULT_DIR, 'lgb_env.json'), 'w')
    json.dump(di, lgb_env)
    lgb_env.close()
    
    del dtrain, X_train; gc.collect()
    
    # load test
    if DATALAKE_TEST_FILE_ID is not None:
        datalake_client = DatalakeClient()
        channel = datalake_client.get_channel(DATALAKE_CHANNEL_ID)
        datalake_file = channel.get_file(DATALAKE_TEST_FILE_ID)
        datalake_file.get_content(cache=True)
        
        csvfile = Path(ABEJA_STORAGE_DIR_PATH, DATALAKE_CHANNEL_ID, DATALAKE_TEST_FILE_ID)
        X_test = pd.read_csv(csvfile, usecols=cols_train)[cols_train]

        is_multi = Parameters.OBJECTIVE.startswith("multi")
        if is_multi:
            pred = np.zeros((len(X_test), Parameters.NUM_CLASS))
        else:
            pred = np.zeros(len(X_test))
        for model in models:
            pred += model.predict(X_test)
        pred /= len(models)
        if is_multi:
            pred = np.argmax(pred, axis=1)
        
        print(pred)
    

if __name__ == '__main__':
    handler(None)
