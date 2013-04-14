import logging
from multiprocessing import Condition
from multiprocessing import Pool
from multiprocessing import sharedctypes
from operator import mul

import numpy as np
import scipy.sparse as sparse

logger = logging.getLogger('grizzly')
logger.setLevel(logging.DEBUG)


def shared_empty_ndarray(shape, dtype):
    try:
        iter(shape)
        n_elems = reduce(mul, shape)
    except TypeError:
        n_elems = shape
    n_bytes = n_elems * np.dtype(dtype).itemsize
    # Use a RawArray instead of an Array because we're going to
    # bypass the lock anyway by having numpy directly use this
    # memory as the backing store.
    mem = sharedctypes.RawArray('c', n_bytes)
    # The `base` attribute of the returned ndarray holds a reference to the
    # sharedctypes array, so we don't need to pass it back.
    # Buffer objects are not writable by default, but Numpy just steals
    # the pointer anyway. We know the array is writable, so poke the flag
    # manually.
    array = np.ndarray(shape=shape, dtype=dtype, buffer=buffer(mem))
    array.flags.writeable = True
    return array


def _shared_dense_ndarray(array):
    shared_array = shared_empty_ndarray(array.shape, array.dtype)
    shared_array[:] = array
    return shared_array


def shared_ndarray_copy(array):
    """Return a copy of ndarray `array` backed by shared memory

    This can be used to share a dense numpy array with multiple processes
    without having to copy the array to all processes.

    The returned ndarray has no intrinsic locks, lock it yourself.
    """
    if not sparse.issparse(array):
        return _shared_dense_ndarray(array)

    # Sparse matrices have different underlying data arrays
    default_params = ('data', 'indices', 'indptr')
    default_asm = lambda data, indices, indptr: (data, indices, indptr)
    format2params = {
        'bsr': (sparse.isspmatrix_bsr, default_params, default_asm,
                sparse.bsr_matrix),
        'coo': (sparse.isspmatrix_coo, ('data', 'row', 'col'),
                lambda data, row, col: (data, (row, col)),
                sparse.coo_matrix),
        'csc': (sparse.isspmatrix_csc, default_params, default_asm,
                sparse.csc_matrix),
        'csr': (sparse.isspmatrix_csr, default_params, default_asm,
                sparse.csr_matrix),
        'dia': (sparse.isspmatrix_dia, ('data', 'offsets'),
                lambda data, offsets: (data, offsets),
                sparse.dia_matrix),
        # dok is unsupported because it's not numpy backed
        # lil is unsupported because it stores objects internally
    }
    for fmt, (pred, params, assembler, ctor) in format2params.iteritems():
        if not pred(array):
            continue
        dense_backings = tuple(_shared_dense_ndarray(getattr(array, name))
                               for name in params)
        return ctor(dense_backings, shape=array.shape, dtype=array.dtype)

    else:
        raise TypeError('Unsupported sparse matrix format')


class MultiprocessCrossValidator(object):
    """Memory-efficient parallel cross-validation across processes


    """
    def __init__(self, estimator_cls, params, X, y, copy=True):
        if copy:
            self.X, self.y = map(shared_ndarray_copy, (X, y))
        else:
            self.X, self.y = X, y
        self.estimator_cls = estimator_cls
        self.params = params

    def async(self, cv_iterator, evaluator, fold_callback=None, n_jobs=None):
        pool = Pool(processes=n_jobs,
                    initializer=self._parallel_initializer,
                    initargs=map(self._ndarray_to_params, (self.X, self.y)))
        args = ({'estimator_cls': self.estimator_cls,
                 'train_indices': train,
                 'test_indices': test,
                 'evaluator': evaluator,
                 'estimator_params': {},
                 'fit_params': {}} for train, test in cv_iterator)
        deferreds = [pool.apply_async(_mpcv_parallel_worker, kwds=kwargs,
                                      callback=fold_callback)
                     for kwargs in args]
        return pool, deferreds

    def __call__(self, cv_iterator, evaluator, fold_callback=None,
                 n_jobs=None):
        """
        """
        condvar = Condition()
        results = []

        def _signal_cb(result):
            condvar.acquire()
            results.append(result)
            condvar.notify()
            condvar.release()
        folds = list(cv_iterator)

        pool, deferreds = self.async(folds, evaluator,
                                     fold_callback=_signal_cb, n_jobs=n_jobs)
        pool.close()
        while len(results) < len(folds):
            condvar.acquire()
            condvar.wait()
            fold_estimator, result = results[-1]
            fold_callback(fold_estimator, result)
            condvar.release()
        pool.join()
        return results

    @staticmethod
    def _ndarray_to_params(array):
        return (array.base, array.shape, array.dtype)

    @staticmethod
    def _ndarray_from_params((mem, shape, dtype)):
        return np.ndarray(shape=shape, dtype=dtype, buffer=buffer(mem))

    @staticmethod
    def _parallel_initializer(Xparams, yparams):
        _mpcv_parallel_worker._parallel_X = \
            MultiprocessCrossValidator._ndarray_from_params(Xparams)
        _mpcv_parallel_worker._parallel_y = \
            MultiprocessCrossValidator._ndarray_from_params(yparams)


def _print_resource_usage():
    from os import getpid
    import resource
    usage = resource.getrusage(resource.RUSAGE_SELF)
    pid = getpid()
    fields = ['ru_utime', 'ru_stime', 'ru_maxrss', 'ru_ixrss',
              'ru_idrss', 'ru_isrss', 'ru_minflt', 'ru_majflt',
              'ru_nswap', 'ru_inblock', 'ru_oublock', 'ru_msgsnd',
              'ru_msgrcv', 'ru_nsignals', 'ru_nvcsw', 'ru_nivcsw']
    lines = ['']
    for field in fields:
        lines.append('%d\t%s\t%s' % (pid, field, str(getattr(usage, field))))
    print '\n'.join(lines)


def _mpcv_parallel_worker(estimator_cls,
                     train_indices, test_indices, evaluator,
                     estimator_params, fit_params):
    #_print_resource_usage()
    X = _mpcv_parallel_worker._parallel_X
    y = _mpcv_parallel_worker._parallel_y
    estimator = estimator_cls(**estimator_params)
    estimator.fit(X[train_indices, :], y[train_indices], **fit_params)
    y_pred = estimator.predict(X[test_indices, :])
    result = evaluator(y[test_indices], y_pred)
    return estimator, result
