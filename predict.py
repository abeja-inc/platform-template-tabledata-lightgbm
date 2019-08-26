import http
import os
import traceback
from io import BytesIO
import json

import pandas as pd
import numpy as np
import lightgbm as lgb


# Configurable, but you don't need to set this. Default is "~/.abeja/.cache" in ABEJA Platform.
ABEJA_STORAGE_DIR_PATH = os.getenv('ABEJA_STORAGE_DIR_PATH')
ABEJA_TRAINING_RESULT_DIR = os.getenv('ABEJA_TRAINING_RESULT_DIR')

with open(os.path.join(ABEJA_TRAINING_RESULT_DIR, 'lgb_env.json')) as f:
    lgb_env = json.load(f)
    NFOLD = lgb_env.get('NFOLD')
    cols_train = lgb_env.get('cols_train')

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
        
        pred = np.zeros(len(X_test))
        for model in models:
            pred += model.predict(X_test)
        pred /= len(models)
        
        print(pred)
        
        return {
            'status_code': http.HTTPStatus.OK,
            'content_type': 'application/json; charset=utf8',
            'content': {'result': pred}
        }
    except Exception as e:
        print(str(e))
        print(traceback.format_exc())
        return {
            'status_code': http.HTTPStatus.INTERNAL_SERVER_ERROR,
            'content_type': 'application/json; charset=utf8',
            'content': {'message': str(e)}
        }


