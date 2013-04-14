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
from sklearn_utils import is_classifier
from sklearn_utils import is_regressor
from sklearn_utils import is_clusterer
import sklearn_utils


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
        # TODO: thread this so that we can interrupt
        self._estimator = self._active_estimator_cls.obj()
        compute_thread = threading.Thread(
            target=GrizzlyModel._estimate_thread,
            kwargs={'model': self, 'completion_callback': callback})
        compute_thread.start()
        # TODO: allocate shared memory arrays for X and y so that each worker
        # proc can read from there. see below.
        # file://localhost/Users/ihaque/oldhome/dox/python-2.7.2-docs-html/library/multiprocessing.html#module-multiprocessing.sharedctypes
        """
            >>> a = np.array([[1,2],[3,4]], dtype=np.uint32)
            >>> import multiprocessing.sharedctypes as sct
            # There's no point to using Array instead of RawArray, because we're
            # going to bypass the lock anyway by accessing through numpy. The
            # call to RawArray is basically just mmap/malloc/shmat.
            >>> ra = sct.RawArray('c', len(a.data))
            >>> ra[:] = a.data
            >>> b = np.ndarray(shape=a.shape, dtype=a.dtype, buffer=buffer(ra))
            >>> a
            array([[1, 2],
                   [3, 4]], dtype=uint32)
            >>> b
            array([[1, 2],
                   [3, 4]], dtype=uint32)
            >>> b.base is ra  #  the new numpy array is backed by shared memory, no new allocation
            True
            >>> ra[0] = '\x0a'
            >>> b
            array([[10,  2],
                   [ 3,  4]], dtype=uint32)
            >>> a
            array([[1, 2],
                   [3, 4]], dtype=uint32)

            b is a copy of a, and is backed by the shared memory we allocated.
            ideally, we should allocate two shmem arrays of the right size
            first and then create numpy aliases for them to load X and y.
            it would be even nicer to not have to have two copies of the data
            around at all (the first one we loaded from ARFF), but this would
            require hacking up the ARFF reader and probably two-pass or
            auto-expanding allocation.
        """
        return

    @staticmethod
    def _estimate_thread(model, completion_callback):
        """main() function of a thread to estimate model parameters
        """
        # Build up the feature matrix from the record arrays given by the ARFF
        # loader
        # TODO: is there a better way to do this?
        # TODO: implement cross-validation
        col_names = [name for idx, name in enumerate(model._meta.names())
                     if idx != model._target_idx]
        X = np.empty((model._data.shape[0], len(col_names)))
        for idx, name in enumerate(col_names):
            X[:, idx] = model._data[name]

        y = model._data[model._meta.names()[model._target_idx]]

        # TODO: should lock the data and the estimator
        model._estimator.fit(X, y)
        logger.info(str(model._estimator))

        # Test
        y_pred = model._estimator.predict(X)

        results = []
        # Compute statistics
        if is_classifier(model._estimator):
            accuracy = metrics.accuracy_score(y, y_pred)
            # TODO: pretty-print confusion matrix
            confusion_matrix = metrics.confusion_matrix(y, y_pred)
            results.append('Accuracy: %f' % accuracy)
            results.append('Confusion matrix:\n%s' % str(confusion_matrix))
        elif is_regressor(model._estimator):
            r2 = metrics.r2_score(y, y_pred)
            mse = metrics.mean_squared_error(y, y_pred)
            mae = metrics.mean_absolute_error(y, y_pred)
            results.append('R^2: %f' % r2)
            results.append('Mean absolute error: %f' % mae)
            results.append('Mean squared error: %f' % mse)

        # TODO: This callback will execute in the compute thread, which is odd
        completion_callback(results)
        return

class GrizzlyApp(App):
    def build(self):
        return GrizzlyController(model=GrizzlyModel(),
                                 classifier_name="N/A")


if __name__ == '__main__':
    GrizzlyApp().run()
