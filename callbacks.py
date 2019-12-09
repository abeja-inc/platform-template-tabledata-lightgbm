from logging import getLogger

from abeja.train.client import Client
from abeja.train.statistics import Statistics as ABEJAStatistics
import numpy as np
from tensorboardX import SummaryWriter

from utils import accuracy_score, roc_auc_score, root_mean_squared_error

logger = getLogger('callback')


class Statistics(object):

    """Trainer extension to report the accumulated results to ABEJA Platform.
    This extension uses the log accumulated by a :class:`LogReport` extension
    to print specified entries of the log in a human-readable format.
    Args:
        entries (list of str): List of keys of observations to print.
        log_report (str or LogReport): Log report to accumulate the
            observations. This is either the name of a LogReport extensions
            registered to the trainer, or a LogReport instance to use
            internally.
    """

    def __init__(self, total_epochs):
        self._total_epochs = total_epochs
        self.client = Client()

    def __call__(self, epoch, train_loss, train_acc, val_loss, val_acc):
        statistics = ABEJAStatistics(num_epochs=self._total_epochs, epoch=epoch)

        statistics.add_stage(ABEJAStatistics.STAGE_TRAIN,
                             train_acc, train_loss)
        statistics.add_stage(ABEJAStatistics.STAGE_VALIDATION,
                             val_acc, val_loss)

        try:
            self.client.update_statistics(statistics)
        except Exception:
            logger.warning('failed to update statistics.')


class TensorBoardCallback(object):
    def __init__(self, statistics: Statistics, writer: SummaryWriter):
        self.statistics = statistics
        self.writer = writer
        self._X_val = None
        self._y_val = None
        self._is_multi = None
        self._num_class = None
        self._evaluate = None

    def __call__(self, env):
        epoch = env.iteration
        epoch_train_loss = 0.0
        epoch_val_acc = 0.0
        epoch_val_loss = 0.0
        if env.evaluation_result_list:
            # FIXME: Currently use the first loss value.
            _, _, epoch_train_loss, _, _ = env.evaluation_result_list[0]

        if self._X_val is not None:
            if self._is_multi:
                pred = np.zeros((len(self._X_val), self._num_class))
            else:
                pred = np.zeros(len(self._X_val))

            models = env.model.boosters
            for model in models:
                pred += model.predict(self._X_val)

            if self._is_multi:
                pred = np.argmax(pred, axis=1)
            else:
                pred /= len(models)
            epoch_val_acc, epoch_val_loss = self._evaluate(self._y_val, pred)

        print('-------------')
        print('epoch {} || Epoch_TRAIN_Loss:{:.4f} || Epoch_VAL_Score]:{:.4f} || Epoch_VAL_Loss]:{:.4f}'.format(
            epoch + 1, epoch_train_loss, epoch_val_acc, epoch_val_loss))
        self.statistics(epoch + 1, epoch_train_loss, None, epoch_val_loss, epoch_val_acc)
        self.writer.add_scalar('main/loss', epoch_train_loss, epoch + 1)
        self.writer.add_scalar('test/acc', epoch_val_acc, epoch + 1)
        self.writer.add_scalar('test/loss', epoch_val_loss, epoch + 1)
        self.writer.flush()

    def set_valid(self, X_val, y_val, is_classification: bool, is_multi: bool, num_class: int):
        self._X_val = X_val
        self._y_val = y_val
        self._is_multi = is_multi
        self._num_class = num_class
        if not is_classification:
            self._evaluate = root_mean_squared_error
        elif is_multi:
            self._evaluate = accuracy_score
        else:
            self._evaluate = roc_auc_score


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
