from collections import OrderedDict
import logging
import threading
#os.environ['KIVY_NO_CONSOLELOG'] = "1"

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
        logger.info('Features: ' + str(self.metadata))

    @property
    def metadata(self):
        if self._meta is None:
            return None
        else:
            return zip(self._meta.names(), self._meta.types())

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
        if var_type == 'nominal' and is_regressor(self._active_estimator_cls):
            errors.append('Attempting to regress on a nominal variable')
        if is_clusterer(self._active_estimator_cls):
            errors.append('Clustering not currently supported')
        var_types = self._meta.types()
        if not all(typ == 'numeric' for ii, typ in enumerate(var_types) if
                   ii != self._target_idx):
            errors.append('Currently can only handle numeric features')

        return len(errors) == 0, errors

    def start_classification(self, callback):
        closure = {'completed_count': 0}

        def cb(estimator, result):
            print "Finished fold %d" % closure['completed_count']
            print "Result", str(result)
            closure['completed_count'] += 1

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

        from sklearn.cross_validation import KFold
        compute_thread = threading.Thread(
            target=GrizzlyModel._estimate_thread,
            kwargs={'estimator_cls': self._active_estimator_cls.obj,
                    'X': X, 'y': y,
                    'folds': KFold(X.shape[0], 10, shuffle=True),
                    'metric': metrics.confusion_matrix,
                    'callback': callback})
        compute_thread.start()
        return

    @staticmethod
    def _estimate_thread(estimator_cls, X, y, folds, metric, callback):
        mpcv = MultiprocessCrossValidator(estimator_cls, None, X, y,
                                          copy=False)
        mpcv(folds, metric, callback)
        return


class GrizzlyApp(App):
    def build(self):
        return GrizzlyController(model=GrizzlyModel(),
                                 classifier_name="N/A")


if __name__ == '__main__':
    GrizzlyApp().run()
