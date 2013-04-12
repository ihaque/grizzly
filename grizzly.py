from collections import OrderedDict
from copy import copy
import logging
import os
#os.environ['KIVY_NO_CONSOLELOG'] = "1"
import kivy
kivy.require('1.6.0')

from kivy.app import App
from kivy.factory import Factory
#from kivy.lang import Builder
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.treeview import TreeView
from kivy.uix.treeview import TreeViewNode
from kivy.uix.popup import Popup
from kivy.properties import ObjectProperty
from kivy.properties import StringProperty

import sklearn_utils


logger = logging.getLogger('grizzly')
logger.setLevel(logging.DEBUG)


class ClickableLabel(Label):
    def __init__(self, onclick=None, *args, **kwargs):
        print "initing clickable label with args, kwargs=", args, kwargs
        super(ClickableLabel, self).__init__(*args, **kwargs)
        self.onclick_callback = onclick

    def on_touch_down(self, touch):
        if (self.onclick_callback is not None and
                not touch.is_mouse_scrolling and
                self.collide_point(touch.x, touch.y)):
            self.onclick_callback(self, touch)
        return super(ClickableLabel, self).on_touch_down(touch)

    def select(self):
        print "selected", self.text
        if self.onclick_callback:
            self.onclick_callback(self, None)


class LoadDialog(FloatLayout):
    filename = ObjectProperty(None)
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)
    path = StringProperty(os.getcwd())


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


class OutputWindowStream(object):
    def __init__(self, controller_obj):
        self.controller = controller_obj

    def write(self, data):
        self.controller.output_text += data

    def flush(self):
        pass


class GrizzlyController(FloatLayout):
    classifier_label = ObjectProperty()
    classifier_name = StringProperty()
    output_text = StringProperty()

    def __init__(self, model, *args, **kwargs):
        super(GrizzlyController, self).__init__(*args, **kwargs)
        self.model = model
        self.current_popup = None
        self.window_logging_handler = logging.StreamHandler(
            OutputWindowStream(self))
        logger.addHandler(self.window_logging_handler)

    def show_popup(self, popup):
        assert self.current_popup is None
        self.current_popup = popup
        popup.open()

    def close_popup(self):
        self.current_popup.dismiss()
        self.current_popup = None

    def choose_classifier(self):
        tree = self._build_classifier_tree(self.select_classifier)

        # Make the tree scrollable inside the ScrollView
        tree.size_hint_y = None
        tree.bind(minimum_height=tree.setter('height'))

        scrollable = ScrollView()
        scrollable.add_widget(tree)
        self.show_popup(Popup(title="Select classifier", content=scrollable))

    def select_classifier(self, instance, touch):
        self.close_popup()
        label = self.model.estimators
        for idx in instance.path:
            label = label[idx]
        self.classifier_name = "%s (%s)" % (label.name, label.fullname)

    def choose_target_variable(self):
        pass

    def load_arff(self, path, filenames):
        assert len(filenames) == 1
        filename = filenames[0]
        logger.info("Loading from %s..." % filename)
        self.model.load_data(filename)
        self.close_popup()

    def onclick_loadfile(self):
        popup = Popup(title="Load ARFF", content=LoadDialog())
        popup.content.load = self.load_arff
        popup.content.cancel = self.close_popup
        self.show_popup(popup)

    def _build_classifier_tree(self, classifier_callback):
        class TreeViewLabel(ClickableLabel, TreeViewNode):
            path = ObjectProperty()

        tv = TreeView(root_options={'text': 'Estimators'})

        def populate_tree(parent, level, index, path):
            obj = level if index is None else level[index]
            if isinstance(obj, _estimator):
                estimator = obj
                label = TreeViewLabel(text=estimator.name,
                                      onclick=classifier_callback)
                label.path = copy(path)
                tv.add_node(label, parent)
            else:
                try:
                    indices = obj.iterkeys()
                except AttributeError:
                    indices = xrange(len(obj))
                if index is not None:
                    group_node = TreeViewLabel(text=index)
                    tv.add_node(group_node, parent)
                else:
                    group_node = None
                for index in indices:
                    populate_tree(group_node, obj, index, path + [index])
            return
        populate_tree(None, self.model.estimators, None, [])
        return tv


class GrizzlyApp(App):
    def build(self):
        return GrizzlyController(model=GrizzlyModel(),
                                 classifier_name="N/A")

Factory.register('LoadDialog', cls=LoadDialog)

if __name__ == '__main__':
    GrizzlyApp().run()
