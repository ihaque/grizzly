import os
#os.environ['KIVY_NO_CONSOLELOG'] = "1"
import kivy
kivy.require('1.6.0')

from kivy.app import App
from kivy.factory import Factory
from kivy.lang import Builder
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
from kivy.properties import ObjectProperty
from kivy.properties import StringProperty

from functools import partial

class LoadDialog(FloatLayout):
    filename = ObjectProperty(None)
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)



class GrizzlyController(FloatLayout):
    classifier_label = ObjectProperty()
    classifier_name = StringProperty()

    def choose_classifier(self):
        self.classifier_name = "Naive Bayes"

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

class GrizzlyApp(App):
    def build(self):
        return GrizzlyController(classifier_name="N/A")

Factory.register('LoadDialog', cls=LoadDialog)

if __name__ == '__main__':
    GrizzlyApp().run()
