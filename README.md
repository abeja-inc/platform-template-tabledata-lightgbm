# platform-template-tabledata-lightgbm
## Environment options
### DataLake
* DATALAKE_CHANNEL_ID: DataLake channel id
* DATALAKE_TRAIN_ID: DataLake item id
* DATALAKE_TEST_ID: DataLake item id

### Features
* USECOLS: Names of features. e.g. var_1,var_2,var_3
* TARGET: Name of target column. This Environment key is required.

### LightGBM
* PARAMS: Parameters to be passed to lgb.train as params. e.g. num_boost_round=200,learning_rate=0.2
