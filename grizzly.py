from collections import OrderedDict
import logging
import multiprocessing
import threading

multiprocessing.log_to_stderr().setLevel(logging.DEBUG)

import numpy as np
from scipy.io.arff import loadarff
import sklearn.metrics as metrics

import kivy
kivy.require('1.6.0')

from kivy.app import App

from kivy_view import GrizzlyController
from sklearn_utils import is_regressor
from sklearn_utils import is_clusterer
import sklearn_utils

from multiprocess import MultiprocessCrossValidator
from multiprocess import shared_empty_ndarray


logger = logging.getLogger('grizzly')
logger.setLevel(logging.DEBUG)


class _estimator(object):
    def __init__(self, modulepath, name):
        self.fullname = "%s.%s" % (modulepath, name)
        self.name = name
        self.obj = sklearn_utils.import_from(modulepath, name)
        self.default_params, self.desc = \
                sklearn_utils.get_params_and_descs(self.obj)


class GrizzlyModel(object):
    def __init__(self):
        self.estimators, self.flat_estimators = self._build_estimator_tree()
        self._data, self._meta = None, None
        self._target_idx = None
        self._active_estimator_cls = None
        self._training_condvar = None
        self._stop_requested = False

    @staticmethod
    def _build_estimator_tree():
        classifiers, regressors, clusterers = sklearn_utils.list_classifiers()
        estimators = OrderedDict()
        flat_estimators = []
        for typ, lst in [('Classifiers', classifiers),
                         ('Regressors', regressors),
                         ('Clusterers', clusterers)]:
            module2name = OrderedDict()
            for modulepath, name in sorted(lst):
                modlist = module2name.setdefault(modulepath, [])
                estimator = _estimator(modulepath, name)
                modlist.append(estimator)
                flat_estimators.append(estimator)
            estimators[typ] = module2name
        return estimators, flat_estimators

    def load_data(self, filename):
        self._data, self._meta = loadarff(filename)
        self._target_idx = None
        logger.info('Loaded %d instances from %s' %
                    (self._data.shape[0], filename))
        logger.info('Features: ' + str(zip(*self.get_variables())))

    def get_variables(self):
        if self._meta is None:
            return None, None
        else:
            return self._meta.names(), self._meta.types()

    def set_target(self, target_name):
        assert target_name in self._meta.names()
        self._target_idx = self._meta.names().index(target_name)
        logger.info('Predicting %s' % target_name)

    def set_estimator(self, estimator):
        assert estimator in self.flat_estimators
        self._active_estimator_cls = estimator

    def ready(self):
        errors = []
        if self._data is None:
            errors.append('No dataset loaded')
        if self._target_idx is None:
            errors.append('Must select a target variable')
        if self._active_estimator_cls is None:
            errors.append('Must select an estimator to use')

        var_type = self._meta.types()[self._target_idx]
        if var_type == 'nominal' and \
                is_regressor(self._active_estimator_cls.obj):
            errors.append('Attempting to regress on a nominal variable')

        if is_clusterer(self._active_estimator_cls.obj):
            errors.append('Clustering not currently supported')

        var_types = self._meta.types()
        if not all(typ == 'numeric' for ii, typ in enumerate(var_types) if
                   ii != self._target_idx):
            errors.append('Currently can only handle numeric features')

        return len(errors) == 0, errors

    @property
    def currently_training(self):
        return self._training_condvar is not None

    def stop_classification(self):
        if not self.currently_training:
            return
        self._training_condvar.acquire()
        self._stop_requested = True
        self._training_condvar.notify()
        self._training_condvar.release()

    def _shmem_feature_matrices(self):
        # Create X and y arrays in shared memory
        col_names = [name for idx, name in enumerate(self._meta.names())
                     if idx != self._target_idx]
        X = shared_empty_ndarray((self._data.shape[0], len(col_names)),
                                 np.float64)
        for idx, name in enumerate(col_names):
            X[:, idx] = self._data[name]

        local_y = self._data[self._meta.names()[self._target_idx]]
        y = shared_empty_ndarray(local_y.shape, local_y.dtype)
        y[:] = local_y

        return X, y

    def start_classification(self, callback):
        if self.currently_training:
            logger.info('Already training!')
            return
        self._stop_requested = False
        self._training_condvar = multiprocessing.Condition()

        from sklearn.cross_validation import KFold
        compute_thread = threading.Thread(
            target=GrizzlyModel._estimate_thread,
            kwargs={'model': self,
                    'estimator_cls': self._active_estimator_cls.obj,
                    'folds': KFold(self._data.shape[0], 10, shuffle=True),
                    'metric': metrics.confusion_matrix,
                    'callback': callback})
        compute_thread.start()
        return

    @staticmethod
    def _estimate_thread(model, estimator_cls, folds, metric, callback):
        X, y = model._shmem_feature_matrices()
        results = []
        completed = 0

        def _asyncresult_callback(result):
            model._training_condvar.acquire()
            results.append(result)
            model._training_condvar.notify()
            model._training_condvar.release()

        # Train the models
        mpcv = MultiprocessCrossValidator(estimator_cls, None, X, y,
                                          copy=False)
        pool, deferreds = mpcv.async(folds, metric, _asyncresult_callback)

        while not model._stop_requested and completed < len(deferreds):
            model._training_condvar.acquire()
            model._training_condvar.wait()
            for fold_estimator, result in results:
                callback(fold_estimator, result)
                completed += 1
            del results[:]
            if model._stop_requested:
                # Calling terminate on a pool causes the process to hang
                # see kivy/kivy/issues/1116
                #pool.terminate()
                logger.warning("Can't stop, won't stop: "
                               "Kivy hangs on terminate")
                model._stop_requested = False
            model._training_condvar.release()

        if not model._stop_requested:
            pool.close()
            pool.join()

        # Clean up
        model._training_condvar = None
        return


class GrizzlyApp(App):
    def build(self):
        return GrizzlyController(model=GrizzlyModel(),
                                 classifier_name="N/A")


if __name__ == '__main__':
    GrizzlyApp().run()
