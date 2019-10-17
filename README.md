# platform-template-tabledata-lightgbm
## Environment options
### DataLake
env name|required|default|description
--------|--------|-------|-----------
DATALAKE_CHANNEL_ID|True|None|DataLake channel id
DATALAKE_TRAIN_FILE_ID|True|None|DataLake file id for training
DATALAKE_VAL_FILE_ID|False|None|DataLake file id for validating

### Features
env name|required|default|description
--------|--------|-------|-----------
INPUT_FIELDS|False|None|Names of features. <br>e.g. var_1,var_2,var_3
LABEL_FIELD|True|None|Name of label column.


### LightGBM
Parameter doc is available on [https://lightgbm.readthedocs.io/en/latest/Parameters.html](https://lightgbm.readthedocs.io/en/latest/Parameters.html)

Try tuning. 
1. num_leaves
1. min_data_in_leaf
1. max_depth

More tuning tips are available on [https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html)

env name|required|default|description
--------|--------|-------|-----------
OBJECTIVE|True|regression|Currently `regression` and `binary` applications are supported. <br>Must be one of `[regression, regression_l1, huber, fair, poisson, quantile, mape, gamma, tweedie, binary, multiclass, multiclassova]`.
IS_CLASSIFICATION|False|True|If `True`, classification, else regression.
BOOSTING|True|gbdt|Must be one of `[gbdt, rf]`
NUM_ITERATIONS|True|100|Number of boosting iterations. constraints: `NUM_ITERATIONS >= 0`
LEARNING_RATE|True|0.05|Shrinkage rate. constraints: `LEARNING_RATE > 0.0`
NUM_LEAVES|True|31|Max number of leaves in one tree. constraints: `NUM_LEAVES > 1`
TREE_LEARNER|True|serial|Must be one of `[serial, feature, data, voting]`.
NUM_THREADS|False|0|Number of threads for LightGBM. `0` means default number of threads in OpenMP.
DEVICE_TYPE|True|cpu|Device for the tree learning, you can use GPU to achieve the faster learning. <br>Must be one of `[cpu, gpu]`.
SEED|False|42|Random seed.
MAX_DEPTH|False|-1|Limit the max depth for tree model. This is used to deal with over-fitting when #data is small. Tree still grows leaf-wise. `<= 0` means no limit.
MIN_DATA_IN_LEAF|False|20|Minimal number of data in one leaf. Can be used to deal with over-fitting. constraints: `MIN_DATA_IN_LEAF >= 0`
MIN_SUM_HESSIAN_IN_LEAF|False|1e-3|Minimal sum hessian in one leaf. Like `MIN_DATA_IN_LEAF`, it can be used to deal with over-fitting. constraints: `MIN_SUM_HESSIAN_IN_LEAF >= 0.0`.
BAGGING_FRACTION|False|1.0|It likes `FEATURE_FRACTION`, but this will randomly select part of data without resampling. constraints: `0.0 < BAGGING_FRACTION <= 1.0`.
POS_BAGGING_FRACTION|False|1.0|Used only in `binary` application. Used for imbalanced binary classification problem, will randomly sample `#pos_samples * pos_bagging_fraction` positive samples in bagging. Should be used together with `NEG_BAGGING_FRACTION`.
NEG_BAGGING_FRACTION|False|1.0|Used only in `binary` application. Same as `POS_BAGGING_FRACTION`.
BAGGING_FREQ|False|0|Frequency for bagging. `0` means disable bagging; `k` means perform bagging at every `k` iteration.
BAGGING_SEED|False|3|Random seed for bagging.
FEATURE_FRACTION|False|1.0|LightGBM will randomly select part of features on each iteration if `FEATURE_FRACTION` smaller than `1.0`. For example, if you set it to `0.8`, LightGBM will select 80% of features before training each tree. constraints: `0.0 < FEATURE_FRACTION <= 1.0`.
FEATURE_FRACTION_SEED|False|2|Random seed for `FEATURE_FRACTION`.
EARLY_STOPPING_ROUNDS|False|10|Will stop training if one metric of one validation data doesn’t improve in last `EARLY_STOPPING_ROUNDS` rounds.
VERBOSITY|False|1|Controls the level of LightGBM’s verbosity. `< 0`: Fatal, `= 0`: Error (Warning), `= 1`: Info, `> 1`: Debug
MAX_BIN|False|255|Max number of bins that feature values will be bucketed in. Small number of bins may reduce training accuracy but may increase general power (deal with over-fitting). constraints: `MAX_BIN > 1`
NUM_CLASS|False|1|Number of classes. Used only in `multi-class` classification applications. constraints: `NUM_CLASS > 0`
METRIC|False|""|Metric. Support multiple metrics, separated by `,`
METRIC_FREQ|False|1|Frequency for metric output. constraints: `METRIC_FREQ > 0`
NFOLD|False|5|Number of folds in CV.
VERBOSE_EVAL|False|None|Whether to display the progress. If None, progress will be displayed when np.ndarray is returned. If True, progress will be displayed at every boosting stage. If int, progress will be displayed at every given verbose_eval boosting stage.
STRATIFIED|False|True|Whether to perform stratified sampling.


## Run on local
Use `requirements-local.txt`.

```
$ pip install -r requirements-local.txt
```

Set environment variables.

| env | type | description |
| --- | --- | --- |
| ABEJA_ORGANIZATION_ID | str | Your organization ID. |
| ABEJA_PLATFORM_USER_ID | str | Your user ID. |
| ABEJA_PLATFORM_PERSONAL_ACCESS_TOKEN | str | Your Access Token. |
| DATASET_ID | str | Dataset ID. |
