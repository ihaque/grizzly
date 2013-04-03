from collections import OrderedDict
from copy import copy
from functools import partial
#import os
#os.environ['KIVY_NO_CONSOLELOG'] = "1"
import kivy
kivy.require('1.6.0')

from kivy.app import App
from kivy.factory import Factory
#from kivy.lang import Builder
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.treeview import TreeView
from kivy.uix.treeview import TreeViewNode
from kivy.uix.popup import Popup
from kivy.properties import ObjectProperty
from kivy.properties import StringProperty

import sklearn_utils


class LoadDialog(FloatLayout):
    filename = ObjectProperty(None)
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


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

    @staticmethod
    def _build_estimator_tree():
        classifiers, regressors, clusterers = sklearn_utils.list_classifiers()
        estimators = OrderedDict()
        for typ, lst in [('classifiers', classifiers),
                         ('regressors', regressors),
                         ('clusterers', clusterers)]:
            module2name = OrderedDict()
            for modulepath, name in sorted(lst):
                modlist = module2name.setdefault(modulepath, [])
                modlist.append(_estimator(modulepath, name))
            estimators[typ] = module2name
        return estimators


class GrizzlyController(FloatLayout):
    classifier_label = ObjectProperty()
    classifier_name = StringProperty()

    def __init__(self, model, *args, **kwargs):
        super(GrizzlyController, self).__init__(*args, **kwargs)
        self.model = model
        self.current_popup = None

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

    def select_classifier(self, instance):
        self.close_popup()
        label = self.model.estimators
        for idx in instance.path:
            label = label[idx]
        self.classifier_name = "%s (%s)" % (label.name, label.fullname)

    def load_arff(self, path, filename, window=None):
        print path
        print filename
        if window:
            window.dismiss()

    def onclick_loadfile(self):
        popup = Popup(title="Load ARFF", content=LoadDialog())
        popup.content.cancel = popup.dismiss
        popup.content.load = partial(self.load_arff, window=popup)
        popup.open()

    def _build_classifier_tree(self, classifier_callback):
        class TreeViewButton(Button, TreeViewNode):
            path = ObjectProperty()
            pass

        class TreeViewLabel(Label, TreeViewNode):
            path = ObjectProperty()

            def on_touch_down(self, touch):
                if self.path and not touch.is_mouse_scrolling:
                    print self.__dict__
                    classifier_callback(self)
                return super(TreeViewLabel, self).on_touch_down(touch)
            pass
        tv = TreeView(root_options={'text': 'Estimators'})

        def populate_tree(parent, level, index, path):
            obj = level[index]
            if isinstance(level[index], _estimator):
                estimator = obj
                button = TreeViewLabel(
                    text=estimator.name,
                    markup=False)
                button.path = copy(path)
                tv.add_node(button, parent)
            else:
                try:
                    indices = obj.iterkeys()
                except AttributeError:
                    indices = xrange(len(obj))
                group_node = TreeViewLabel(text=index)
                tv.add_node(group_node, parent)
                for index in indices:
                    populate_tree(group_node, obj, index, path + [index])
            return
        populate_tree(None,
                      {'estimators': self.model.estimators},
                      'estimators', [])
        return tv


class GrizzlyApp(App):
    def build(self):
        return GrizzlyController(model=GrizzlyModel(),
                                 classifier_name="N/A")

Factory.register('LoadDialog', cls=LoadDialog)

if __name__ == '__main__':
    GrizzlyApp().run()
