# platform-template-tabledata-lightgbm
## Environment options
### DataLake
env name|required|default|description
--------|--------|-------|-----------
DATALAKE_CHANNEL_ID|True|None|DataLake channel id
DATALAKE_TRAIN_ITEM_ID|True|None|DataLake item id
DATALAKE_TEST_ITEM_ID|False|None|DataLake item id

### Features
env name|required|default|description
--------|--------|-------|-----------
INPUT_FIELDS|False|None|Names of features. <br>e.g. var_1,var_2,var_3
TARGET_FIELD|True|None|Name of target column.


### LightGBM
env name|required|default|description
--------|--------|-------|-----------
PARAMS|False|dict|To be passed to lgb.cv as params. <br>e.g. num_boost_round=200,learning_rate=0.2 <br>Refer to https://lightgbm.readthedocs.io/en/latest/Parameters.html
NFOLD|False|5|To be passed to nfold.
EARLY_STOPPING_ROUNDS|False|None|To be passed to early_stopping_rounds
VERBOSE_EVAL|False|None|To be passed to verbose_eval
STRATIFIED|False|None|To be passed to stratified

