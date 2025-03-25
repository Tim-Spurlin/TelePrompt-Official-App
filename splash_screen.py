# splash_screen.py

import os
from kivy.uix.screenmanager import Screen
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.properties import NumericProperty
from kivy.graphics import PushMatrix, PopMatrix, Rotate

class SplashScreen(Screen):
    """
    A simple Splash Screen that displays a rotating logo for a few seconds,
    then transitions to the main screen.
    """

    angle = NumericProperty(0)  # We'll rotate the image using this property

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # 1) Build the relative path to your logo folder.
        #    We'll assume your structure is something like:
        #    teleprompt/
        #       ├─ main.py
        #       ├─ splash_screen.py
        #       └─ logo/
        #           └─ my_logo.png
        #
        # This code uses the folder "logo" next to this file.
        script_dir = os.path.dirname(__file__)
        logo_path = os.path.join(script_dir, "logo", "logo.png")  # <--- rename if your file is "logo.png"

        # 2) Create an Image widget for your logo
        self.logo = Image(
            source=logo_path,
            size_hint=(None, None),
            size=(200, 200),
            pos_hint={"center_x": 0.5, "center_y": 0.5}
        )
        self.add_widget(self.logo)

        # 3) Start a repeating clock event that increments self.angle (spin effect)
        Clock.schedule_interval(self.rotate_logo, 1/60.0)  # ~60 FPS

        # 4) Optionally schedule a time to end the splash
        Clock.schedule_once(self.finish_loading, 5.0)

    def rotate_logo(self, dt):
        """
        Called every frame to rotate the logo.
        We just increment the angle and apply a Rotate transform.
        """
        self.angle += 2  # degrees per frame (adjust to taste)
        self.logo.canvas.before.clear()
        with self.logo.canvas.before:
            PushMatrix()
            Rotate(angle=self.angle, origin=self.logo.center)

    def finish_loading(self, *args):
        """
        Called once the "loading" is done.
        Switch to the main screen, stop rotating.
        """
        # Stop spinning
        Clock.unschedule(self.rotate_logo)
        # Return to normal state
        self.logo.canvas.before.clear()

        # Switch to main screen (the name "main" must match your main screen's name in ScreenManager)
        if self.manager is not None:
            self.manager.current = "main"