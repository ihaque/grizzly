from collections import namedtuple
from copy import copy
import logging
import os

from kivy.factory import Factory

from kivy.properties import ObjectProperty
from kivy.properties import StringProperty

from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
from kivy.uix.treeview import TreeView
from kivy.uix.treeview import TreeViewNode

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


class TreeViewLabel(ClickableLabel, TreeViewNode):
    path = ObjectProperty()


class LoadDialog(FloatLayout):
    filename = ObjectProperty(None)
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)
    path = StringProperty(os.getcwd())


class OutputWindowStream(object):
    def __init__(self, controller_obj):
        self.controller = controller_obj

    def write(self, data):
        self.controller.output_text += data

    def flush(self):
        pass


def create_scrollable_treeview(nested_dicts, onclick, root_label):
    tv = TreeView(root_options={'text': root_label})

    def populate_tree(parent, level, index, path):
        obj = level if index is None else level[index]
        if hasattr(obj, 'name'):
            label = TreeViewLabel(text=obj.name,
                                  onclick=onclick)
            label.path = copy(path)
            tv.add_node(label, parent)
        else:
            try:
                indices = obj.iterkeys()
            except AttributeError:
                indices = xrange(len(obj))
            # This `if` is a hack to deal with the root of the given tree
            if index is None:
                group_node = None
            else:
                group_node = TreeViewLabel(text=index)
                tv.add_node(group_node, parent)
            for index in indices:
                populate_tree(group_node, obj, index, path + [index])

    populate_tree(None, nested_dicts, None, [])

    # Make the tree scrollable inside the ScrollView
    tv.size_hint_y = None
    tv.bind(minimum_height=tv.setter('height'))
    scrollable = ScrollView()
    scrollable.add_widget(tv)
    return scrollable


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
        self.show_popup(
            Popup(title="Select classifier",
                  content=create_scrollable_treeview(self.model.estimators,
                                                     self.select_classifier,
                                                     'Estimators')))

    def select_classifier(self, instance, touch):
        self.close_popup()
        label = self.model.estimators
        for idx in instance.path:
            label = label[idx]
        self.classifier_name = "%s (%s)" % (label.name, label.fullname)
        self.model.set_estimator(label)

    def choose_target_variable(self):
        _vartype = namedtuple('vartype', ('name', ))
        var_labels = [_vartype('%s (%s)' % (fname, ftype)) for fname, ftype in
                      self.model.metadata]
        self.show_popup(
            Popup(title='Select target variable',
                content=create_scrollable_treeview(var_labels,
                                                   self.select_target_variable,
                                                   'Variables')))

    def select_target_variable(self, instance, touch):
        self.close_popup()
        index = instance.path[0]
        self.model.set_target(self.model.metadata[index][0])
        # TODO: update display text

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

    def start_classification(self):
        ready, messages = self.model.ready()
        if not ready:
            logger.error('Cannot start classification:')
            logger.error('\n'.join(messages))
            return
        self.model.start_classification(self.show_classification_results)

    def show_classification_results(self, __, results):
        logger.info('\n'.join(map(str,results)))


Factory.register('LoadDialog', cls=LoadDialog)
