import os


def get_env_var(key, converter, default=None):
    value = os.getenv(key)
    if value:
        return converter(value)
    return default


def get_env_var_bool(key, default: bool):
    value = os.getenv(key)
    if value:
        return value.lower() == 'true'
    return default


def get_env_var_csv(key, converter):
    value = os.getenv(key)
    if value:
        return list(set([converter(x.strip()) for x in value.split(',')]))
    return list()


def get_env_var_required(key, converter):
    value = get_env_var(key, converter)
    if value is None:
        raise Exception(f'"{key}"" is required.')
    return value


def get_env_var_validate(key, converter, default=None, min_=None, max_=None, list_=None):
    value = get_env_var(key, converter, default)
    if value:
        if min_ and value < min_:
            raise Exception(f'"{key}" must be "{min_} =< x =< {max_}"')
        if max_ and value > max_:
            raise Exception(f'"{key}" must be "{min_} =< x =< {max_}"')
        if list_ and value not in list_:
            raise Exception(f'"{key}" must be one of [{",".join(list_)}]')
    return value


def get_env_var_metric(key, list_):
    values = os.getenv(key, '').split(',')
    for value in values:
        if value not in list_:
            raise Exception(f'"{key}" must be one of [{",".join(list_)}]')
    rtn = ",".join(values)
    if rtn:
        return rtn
    return None


class Parameters:
    """Parameter class
    parameter name must be consist of upper case characters.
    """
    DATALAKE_CHANNEL_ID = get_env_var_required('DATALAKE_CHANNEL_ID', str)
    DATALAKE_TRAIN_FILE_ID = get_env_var_required('DATALAKE_TRAIN_FILE_ID', str)
    DATALAKE_TEST_FILE_ID = os.getenv('DATALAKE_TEST_FILE_ID')

    INPUT_FIELDS = get_env_var_csv('INPUT_FIELDS', str)
    LABEL_FIELD = get_env_var_required('LABEL_FIELD', str)

    # Core Parameters
    _OBJECTIVE_LIST = [
        "regression", "regression_l1", "huber", "fair", "poisson", "quantile", "mape",
        "gamma", "tweedie", "binary",
        # "multiclass", "multiclassova", "cross_entropy", "cross_entropy_lambda", "lambdarank"  # TODO
    ]
    OBJECTIVE = get_env_var_validate('OBJECTIVE', str, "regression", list_=_OBJECTIVE_LIST)
    _BOOSTING_LIST = [
        "gbdt", "rf",
        # "dart", "goss"
    ]
    BOOSTING = get_env_var_validate('BOOSTING', str, "gbdt", list_=_BOOSTING_LIST)
    NUM_ITERATIONS = get_env_var_validate('NUM_ITERATIONS', int, 100, min_=0)
    LEARNING_RATE = get_env_var_validate('LEARNING_RATE', float, 0.05, min_=1e-10)
    NUM_LEAVES = get_env_var_validate('NUM_LEAVES', int, 31, min_=2)
    _TREE_LEARNER_LIST = ["serial", "feature", "data", "voting"]
    TREE_LEARNER = get_env_var_validate('TREE_LEARNER', str, "serial", list_=_TREE_LEARNER_LIST)
    NUM_THREADS = get_env_var_validate('NUM_THREADS', int, 0, min_=0)
    DEVICE_TYPE = get_env_var_validate('DEVICE_TYPE', str, "cpu", list_=["cpu", "gpu"])
    SEED = get_env_var('SEED', int, 42)

    # Learning Control Parameters
    MAX_DEPTH = get_env_var('MAX_DEPTH', int, -1)
    MIN_DATA_IN_LEAF = get_env_var_validate('MIN_DATA_IN_LEAF', int, 20, min_=0)
    MIN_SUM_HESSIAN_IN_LEAF = get_env_var_validate('MIN_SUM_HESSIAN_IN_LEAF', float, 1e-3, min_=1e-10)
    BAGGING_FRACTION = get_env_var_validate('BAGGING_FRACTION', float, 1, min_=1e-10, max_=1)
    POS_BAGGING_FRACTION = get_env_var_validate('POS_BAGGING_FRACTION', float, 1, min_=1e-10, max_=1)
    NEG_BAGGING_FRACTION = get_env_var_validate('NEG_BAGGING_FRACTION', float, 1, min_=1e-10, max_=1)
    BAGGING_FREQ = get_env_var('BAGGING_FREQ', int, 0)
    BAGGING_SEED = get_env_var('BAGGING_SEED', int, 3)
    FEATURE_FRACTION = get_env_var_validate('FEATURE_FRACTION', float, 1, min_=1e-10, max_=1)
    FEATURE_FRACTION_SEED = get_env_var('FEATURE_FRACTION_SEED', int, 2)
    EARLY_STOPPING_ROUNDS = get_env_var('EARLY_STOPPING_ROUNDS', int, 10)

    # IO Parameters
    VERBOSITY = get_env_var('VERBOSITY', int, 1)
    MAX_BIN = get_env_var_validate('MAX_BIN', int, 255, min_=2)

    # Metric Parameters
    _METRIC_LIST = [
        "", "None", "l1", "l2", "rmse", "quantile", "mape", "huber", "fair",
        "poisson", "gamma", "gamma_deviance", "tweedie", "ndcg", "map", "auc",
        "binary_logloss", "binary_error",
        # "multi_logloss", "multi_error",
        "cross_entropy", "cross_entropy_lambda", "kullback_leibler"
    ]
    METRIC = get_env_var_metric('METRIC', _METRIC_LIST)
    METRIC_FREQ = get_env_var_validate('METRIC_FREQ', int, 1, min_=1)

    # Other Parameters
    VERBOSE_EVAL = get_env_var('VERBOSE_EVAL', int)
    STRATIFIED = get_env_var_bool('STRATIFIED', True)
    NFOLD = get_env_var_validate('NFOLD', int, default=5, min_=2, max_=None)

    # ABEJA Platform environment variables
    ABEJA_STORAGE_DIR_PATH = os.getenv('ABEJA_STORAGE_DIR_PATH', '~/.abeja/.cache')
    ABEJA_TRAINING_RESULT_DIR = os.getenv('ABEJA_TRAINING_RESULT_DIR', 'abejainc_training_result')

    @classmethod
    def as_dict(cls):
        rtn = {
            k: v for k, v in cls.__dict__.items()
            if k.isupper() and not k.startswith("_")
        }
        if rtn["BOOSTING"] == "rf":
            if rtn["BAGGING_FREQ"] == 0:
                rtn["BAGGING_FREQ"] = 5
            if rtn["BAGGING_FRACTION"] == 1:
                rtn["BAGGING_FRACTION"] = 0.5
        if rtn["OBJECTIVE"] != "binary":
            rtn.pop("POS_BAGGING_FRACTION", None)
            rtn.pop("NEG_BAGGING_FRACTION", None)
        return rtn

    @classmethod
    def as_params(cls):
        rtn = {
            k.lower(): v for k, v in cls.as_dict().items()
        }
        for key in [
            "input_fields", "metric", "abeja_storage_dir_path", "stratified",
            "nfold", "datalake_train_file_id", "label_field", "abeja_training_result_dir",
            "datalake_channel_id"
        ]:
            rtn.pop(key, None)
        return rtn
