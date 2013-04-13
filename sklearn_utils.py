from sklearn.base import ClassifierMixin
from sklearn.base import ClusterMixin
from sklearn.base import RegressorMixin

_sklearn_base = 'sklearn'
_sklearn_estimator_modules = [
    # Clustering
    'cluster',
    # Classification and Regression
    'gaussian_process',
    'hmm',
    'isotonic',
    'lda',
    'linear_model',
    'mixture',
    'naive_bayes',
    'neighbors',
    'pls',
    'qda',
    'svm',
    'tree',
]


def list_classifiers():
    from inspect import getmembers
    from inspect import isabstract
    from inspect import isclass
    cls = []
    reg = []
    clus = []
    other = []
    for modulename in _sklearn_estimator_modules:
        modulepath = '%s.%s' % (_sklearn_base, modulename)
        module = __import__(modulepath, fromlist=[modulename])
        for name, obj in getmembers(module, lambda obj: isclass(obj) and
                                                        not isabstract(obj)):
            obj = getattr(module, name)
            if obj is ClassifierMixin or obj is RegressorMixin:
                continue
            if issubclass(obj, ClassifierMixin):
                cls.append((modulepath, name))
            elif issubclass(obj, RegressorMixin):
                reg.append((modulepath, name))
            elif issubclass(obj, ClusterMixin):
                clus.append((modulepath, name))
            else:
                other.append((modulepath, name))
    return cls, reg, clus


def _parse_docstring_for_params(docstring):
    BEFORE_PARAM = 0
    PARAM_HEADER = 1
    IN_PARAM = 2
    state = BEFORE_PARAM
    param_text = []
    for line in docstring.split('\n'):
        if state == BEFORE_PARAM:
            if 'Parameters' in line:
                state = PARAM_HEADER
                param_text.append(line)
        elif state == PARAM_HEADER:
            state = IN_PARAM
            param_text.append(line)
        elif state == IN_PARAM:
            sline = line.strip()
            if sline and all(ch == '-' for ch in sline):
                # Reached the next section; pop the prev line.
                param_text.pop()
                break
            else:
                param_text.append(line)
        else:
            assert False, "Bad state in parser"
    return '\n'.join(param_text)


def get_params_and_descs(estimator):
    desc = _parse_docstring_for_params(estimator.__doc__)
    default_params = estimator().get_params()
    return default_params, desc


def import_from(path, name):
    return getattr(__import__(path, fromlist=[name]), name)


def _is_subclass_or_instance(obj, cls):
    try:
        return issubclass(obj, cls)
    except TypeError:
        return isinstance(obj, cls)


def is_classifier(estimator):
    return _is_subclass_or_instance(estimator, ClassifierMixin)


def is_regressor(estimator):
    return _is_subclass_or_instance(estimator, RegressorMixin)


def is_clusterer(estimator):
    return _is_subclass_or_instance(estimator, ClusterMixin)
