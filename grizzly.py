from collections import OrderedDict
import logging
#os.environ['KIVY_NO_CONSOLELOG'] = "1"
import kivy
kivy.require('1.6.0')

from kivy.app import App

from kivy_view import GrizzlyController
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
        self.estimators = self._build_estimator_tree()
        self._data, self._meta = None, None
        self._target = None

    @staticmethod
    def _build_estimator_tree():
        classifiers, regressors, clusterers = sklearn_utils.list_classifiers()
        estimators = OrderedDict()
        for typ, lst in [('Classifiers', classifiers),
                         ('Regressors', regressors),
                         ('Clusterers', clusterers)]:
            module2name = OrderedDict()
            for modulepath, name in sorted(lst):
                modlist = module2name.setdefault(modulepath, [])
                modlist.append(_estimator(modulepath, name))
            estimators[typ] = module2name
        return estimators

    def load_data(self, filename):
        from scipy.io.arff import loadarff
        self._data, self._meta = loadarff(filename)
        self._target = None
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
        self._target = target_name


class GrizzlyApp(App):
    def build(self):
        return GrizzlyController(model=GrizzlyModel(),
                                 classifier_name="N/A")


if __name__ == '__main__':
    GrizzlyApp().run()
