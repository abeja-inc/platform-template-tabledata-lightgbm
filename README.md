# platform-template-tabledata-lightgbm
## Environment options
### DataLake
* DATALAKE_CHANNEL_ID: DataLake channel id
* DATALAKE_TRAIN_ITEM_ID: DataLake item id
* DATALAKE_TEST_ITEM_ID: DataLake item id

### Features
* INPUT_FIELDS: Names of features. e.g. var_1,var_2,var_3
* TARGET_FIELD: Name of target column. This Environment key is required.

### LightGBM
* PARAMS: Parameters to be passed to lgb.cv as params. e.g. num_boost_round=200,learning_rate=0.2
Refet to https://lightgbm.readthedocs.io/en/latest/Parameters.html
* NFOLD: to be passed to nfold
* EARLY_STOPPING_ROUNDS: to be passed to early_stopping_rounds
* VERBOSE_EVAL: to be passed to verbose_eval
* STRATIFIED: to be passed to stratified
