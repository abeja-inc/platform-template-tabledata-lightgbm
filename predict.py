import http
import os
import traceback
from io import BytesIO
import json
from pathlib import Path

import pandas as pd
import numpy as np
import lightgbm as lgb


ABEJA_STORAGE_DIR_PATH = os.getenv('ABEJA_STORAGE_DIR_PATH', '~/.abeja/.cache')
ABEJA_TRAINING_RESULT_DIR = os.getenv('ABEJA_TRAINING_RESULT_DIR', 'abejainc_training_result')
Path(ABEJA_TRAINING_RESULT_DIR).mkdir(exist_ok=True)

with open(os.path.join(ABEJA_TRAINING_RESULT_DIR, 'lgb_env.json')) as f:
    lgb_env = json.load(f)
    NFOLD = lgb_env.get('NFOLD')
    cols_train = lgb_env.get('cols_train')
    OBJECTIVE = lgb_env.get('OBJECTIVE')
    IS_MULTI = OBJECTIVE.startswith("multi")
    NUM_CLASS = lgb_env.get('NUM_CLASS', 1)

models = []
for i in range(NFOLD):
    model = lgb.Booster(model_file=os.path.join(ABEJA_TRAINING_RESULT_DIR, f'model_{i}.txt'))
    models.append(model)


def handler(request, context):
    print('Start predict handler.')
    if 'http_method' not in request:
        message = 'Error: Support only "abeja/all-cpu:19.04" or "abeja/all-gpu:19.04".'
        print(message)
        return {
            'status_code': http.HTTPStatus.BAD_REQUEST,
            'content_type': 'application/json; charset=utf8',
            'content': {'message': message}
        }

    try:
        data = request.read()
        csvfile = BytesIO(data)
        
        X_test = pd.read_csv(csvfile, usecols=cols_train)[cols_train]

        if IS_MULTI:
            pred = np.zeros((len(X_test), NUM_CLASS))
        else:
            pred = np.zeros(len(X_test))
        for model in models:
            pred += model.predict(X_test)
        pred /= len(models)

        if OBJECTIVE == 'binary':
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
        elif IS_MULTI:
            pred = np.argmax(pred, axis=1)
        
        print(pred)
        
        X_test['pred'] = pred
        
        return {
            'status_code': http.HTTPStatus.OK,
            'content_type': 'application/json; charset=utf8',
            'content': {'result': X_test.values, 
                        'field': X_test.columns.tolist()}
        }
    except Exception as e:
        print(str(e))
        print(traceback.format_exc())
        return {
            'status_code': http.HTTPStatus.INTERNAL_SERVER_ERROR,
            'content_type': 'application/json; charset=utf8',
            'content': {'message': str(e)}
        }


