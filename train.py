# coding: utf-8
# Template: 

import os
import json
from pathlib import Path

import lightgbm as lgb
from tensorboardX import SummaryWriter

from callbacks import Statistics, TensorBoardCallback
from data_loader import train_data_loader
from parameters import Parameters


ABEJA_STORAGE_DIR_PATH = os.getenv('ABEJA_STORAGE_DIR_PATH', '~/.abeja/.cache')
ABEJA_TRAINING_RESULT_DIR = os.getenv('ABEJA_TRAINING_RESULT_DIR', 'abejainc_training_result')
Path(ABEJA_TRAINING_RESULT_DIR).mkdir(exist_ok=True)

DATALAKE_CHANNEL_ID = Parameters.DATALAKE_CHANNEL_ID
DATALAKE_TRAIN_FILE_ID = Parameters.DATALAKE_TRAIN_FILE_ID
DATALAKE_VAL_FILE_ID = Parameters.DATALAKE_VAL_FILE_ID
INPUT_FIELDS = Parameters.INPUT_FIELDS
LABEL_FIELD = Parameters.LABEL_FIELD
PARAMS = Parameters.as_params()

STRATIFIED = True if Parameters.STRATIFIED and Parameters.IS_CLASSIFICATION else False
IS_MULTI = Parameters.OBJECTIVE.startswith("multi")

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

    X_train, y_train, cols_train = train_data_loader(
        DATALAKE_CHANNEL_ID, DATALAKE_TRAIN_FILE_ID, LABEL_FIELD, INPUT_FIELDS)
    dtrain = lgb.Dataset(X_train, y_train)

    if DATALAKE_VAL_FILE_ID:
        X_val, y_val, _ = train_data_loader(
            DATALAKE_CHANNEL_ID, DATALAKE_VAL_FILE_ID, LABEL_FIELD, INPUT_FIELDS)
    else:
        X_val, y_val = None, None

    extraction_cb = ModelExtractionCallback()
    tensorboard_cb = TensorBoardCallback(statistics, writer)
    tensorboard_cb.set_valid(X_val, y_val, Parameters.IS_CLASSIFICATION, IS_MULTI, Parameters.NUM_CLASS)
    callbacks = [extraction_cb, tensorboard_cb,]
    
    lgb.cv(PARAMS, dtrain, nfold=Parameters.NFOLD,
           early_stopping_rounds=Parameters.EARLY_STOPPING_ROUNDS,
           verbose_eval=Parameters.VERBOSE_EVAL,
           stratified=STRATIFIED,
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


if __name__ == '__main__':
    handler(None)
