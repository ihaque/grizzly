import os
#os.environ['KIVY_NO_CONSOLELOG'] = "1"
import kivy
kivy.require('1.6.0')

from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import ObjectProperty
from kivy.properties import StringProperty

class GrizzlyController(FloatLayout):
    classifier_label = ObjectProperty()
    classifier_name = StringProperty()

    def choose_classifier(self):
        self.classifier_name = "Naive Bayes"


class GrizzlyApp(App):
    def build(self):
        return GrizzlyController(classifier_name="N/A")


if __name__ == '__main__':
    GrizzlyApp().run()
