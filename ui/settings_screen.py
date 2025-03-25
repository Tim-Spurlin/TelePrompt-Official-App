import os
import json
from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty, StringProperty
from embedding_manager import embedding_manager  # Import the shared instance
from kivy.graphics import Color, Line
import time
from kivy.clock import Clock
from pathlib import Path
import asyncio

# Use an environment variable with a fallback to the relative path
config_dir = Path(Path(__file__).parent.parent / "config")
config_file = config_dir / "hazel-sky-450118-p6-0ffc677fdd97.json"

if not config_file.exists():
    raise FileNotFoundError(f"Config file not found: {config_file}")

with open(config_file, "r") as f:
    config_data = json.load(f)

class SettingsScreen(Screen):
    """
    A Settings screen with:
      - Top label showing "Active Preset: X" or "No Preset Active"
      - 5 "Preset X" rows in the middle, each row has a half-width button (centered).
      - "Back to Main" pinned at bottom
      - Each preset popup with "Rename", "Delete Preset", doc list, Activate/Deactivate, etc.
      - Renaming also renames/creates the folder "presets/<NewName>".
      - Deleting resets it to "Preset X" with empty docs, removing the old folder and FAISS index.
      - Buttons scale proportionally when the window is resized.
    """
    app_instance = ObjectProperty(None)
    calibration_value = StringProperty("Not Calibrated")

    def __init__(self, config, **kwargs):
        self.config = config
        self.app_instance = kwargs.get('app_instance')
        super().__init__(**kwargs)

        self.layout = BoxLayout(orientation='vertical', spacing=10, padding=10)

        # Top label
        self.active_preset_label = Label(
            text=self._get_active_preset_text(),
            size_hint=(1, 0.1),
            font_size='26sp',
            color=(1, 1, 1, 1),
            halign='center',
            valign='middle'
        )
        self.active_preset_label.bind(
            size=lambda lbl, val: setattr(lbl, 'text_size', val)
        )
        self.layout.add_widget(self.active_preset_label)

        # Middle area with 5 preset rows
        self.middle_layout = BoxLayout(
            orientation='vertical',
            spacing=25,     # <--- ADJUST SPACING BETWEEN ROWS HERE
            size_hint=(1, 0.7)
        )
        self.layout.add_widget(self.middle_layout)

        self.preset_buttons = {}

        # Bottom bar
        bottom_bar = BoxLayout(orientation='horizontal', size_hint=(1, 0.1), spacing=10)
        back_btn = Button(
            text="Back to Main",
            size_hint=(None, None),
            size=(275, 60),
            font_size='22sp',
            background_color=(0.2, 0.6, 1, 1),
            color=(1, 1, 1, 1)
        )
        back_btn.bind(on_press=self.go_back)
        bottom_bar.add_widget(back_btn)
        bottom_bar.add_widget(Widget())
        self.layout.add_widget(bottom_bar)

        self.add_widget(self.layout)
        self.update_preset_button_highlight()

        # Instead, create the timer label reference, but do not add it:
        self.timer_label = Label(text="00:00:000", font_size="30sp", size_hint=(1, 0.1), halign="center")

        # Build the list from config (see rebuild_preset_list method below)
        self.rebuild_preset_list()

    def open_preset_popup(self, preset_name):
        # 1) Main vertical layout for the popup
        popup_layout = BoxLayout(orientation='vertical', spacing=10, padding=(10, 0, 10, 10), size_hint=(1, 1))

        # -------------------------------------------------------------
        # TOP BAR: Only Delete Preset remains at the top
        # -------------------------------------------------------------
        top_bar = BoxLayout(orientation='horizontal', size_hint_y=None, height=60, spacing=10)
        delete_btn = Button(
            text="Delete Preset",
            size_hint=(None, None),
            size=(240, 50),           # <--- Adjust button size here
            font_size='20sp'          # <--- Adjust button font size here
        )
        delete_btn.bind(on_press=lambda x: self.delete_preset(preset_name))
        top_bar.add_widget(delete_btn)
        popup_layout.add_widget(top_bar)

        # -------------------------------------------------------------
        # SCROLL AREA FOR DOCUMENTS
        # -------------------------------------------------------------
        scroll = ScrollView(size_hint=(1, 1))
        docs_box = BoxLayout(orientation='vertical', spacing=5, size_hint_y=None)
        docs_box.bind(minimum_height=docs_box.setter('height'))
        scroll.add_widget(docs_box)
        presets = self.app_instance.teleprompt_config.setdefault("presets", {})
        if preset_name not in presets:
            presets[preset_name] = {"parsed_documents": []}
        for doc_info in presets[preset_name].get("parsed_documents", []):
            doc_label = Label(
                text=os.path.basename(doc_info["filename"]),
                size_hint_y=None,
                height=30,
                font_size='18sp',
                color=(1, 1, 1, 1)
            )
            docs_box.add_widget(doc_label)
        popup_layout.add_widget(scroll)

        # -------------------------------------------------------------
        # BIG, ROUNDED ACTIVATE/DEACTIVATE BUTTON (CENTERED)
        # -------------------------------------------------------------
        from kivy.graphics import Color, RoundedRectangle

        activate_btn = Button(
            text="Activate",
            size_hint=(.75, None),  # full width in the popup
            height=100,             # <--- Adjust button height here
            font_size='24sp',       # <--- Adjust font size here
            background_normal='',   # ensures custom drawing is visible
            background_color=(1,1,1,0),  # transparent background
            bold=True,
            halign='center',
        )
        if self.app_instance.selected_preset == preset_name:
            activate_btn.text = "Deactivate"

        def toggle_preset(_btn):
            if self.app_instance.selected_preset == preset_name:
                # Deactivate
                self.app_instance.selected_preset = None
                activate_btn.text = "Activate"
            else:
                # Activate
                self.app_instance.selected_preset = preset_name
                activate_btn.text = "Deactivate"
            self.app_instance.save_settings()
            self.update_active_preset_label()
            self.update_preset_button_highlight()
             # Force the color to redraw immediately
            draw_activate_button(None)

        activate_btn.bind(on_press=toggle_preset)

        # Draw a full rounded rectangle behind the button in red (inactive) or green (active)
        def draw_activate_button(*args):
            activate_btn.canvas.before.clear()
            with activate_btn.canvas.before:
                if self.app_instance.selected_preset == preset_name:
                    # If active, fill the entire button green
                    Color(0, 1, 0, 1)
                else:
                    # If inactive, fill the entire button red
                    Color(1, 0, 0, 1)
                # Rounded rectangle with adjustable corner radius
                RoundedRectangle(pos=activate_btn.pos, size=activate_btn.size, radius=[20,])
        activate_btn.bind(pos=draw_activate_button, size=draw_activate_button)
        
        # Create a horizontal BoxLayout for the activate button row
        activate_row = BoxLayout(orientation='horizontal', size_hint=(1, None), height=80, spacing=10) # or whatever height you want
       
        # Left spacer -> Activate Button -> Right spacer
        activate_row.add_widget(Widget(size_hint_x=1))
        activate_row.add_widget(activate_btn)
        activate_row.add_widget(Widget(size_hint_x=1))

        # Finally, add the row to your popup layout
        popup_layout.add_widget(activate_row)

        # -------------------------------------------------------------
        # BOTTOM BAR: Four Evenly Distributed Buttons (Back, Rename, Upload Files, Confirm)
        # -------------------------------------------------------------
        bottom_bar = BoxLayout(orientation='horizontal', spacing=10, size_hint=(1, None), height=60)
        back_btn = Button(
            text="Back",
            size_hint=(1, 1),
            font_size='18sp'
        )
        rename_btn = Button(
            text="Rename",
            size_hint=(1, 1),
            font_size='18sp'
        )
        upload_btn = Button(
            text="Upload Files",
            size_hint=(1, 1),
            font_size='18sp'
        )
        confirm_btn = Button(
            text="Confirm",
            size_hint=(1, 1),
            font_size='18sp'
        )
        def close_popup(_instance):
            popup.dismiss()
        back_btn.bind(on_press=close_popup)
        rename_btn.bind(on_press=lambda _: self.rename_preset(preset_name))
        confirm_btn.bind(on_press=close_popup)
        upload_btn.bind(on_release=lambda x: self.open_file_manager_for_preset(preset_name))
        bottom_bar.add_widget(back_btn)
        bottom_bar.add_widget(rename_btn)
        bottom_bar.add_widget(upload_btn)
        bottom_bar.add_widget(confirm_btn)
        popup_layout.add_widget(bottom_bar)

        # -------------------------------------------------------------
        # BUILD & OPEN THE POPUP
        # -------------------------------------------------------------
        popup = Popup(
            title=preset_name,
            content=popup_layout,
            size_hint=(0.9, 0.9),
            auto_dismiss=False,
            background_color=(1, 1, 1, 0.8)  # 0.8 alpha for slight transparency
        )
        popup.open()

        # Load existing index if needed using the shared instance
        index_path = os.path.join("presets", preset_name, "index.faiss")
        doc_map_path = os.path.join("presets", preset_name, "doc_map.npy")
        if os.path.exists(index_path) and os.path.exists(doc_map_path):
            try:
                embedding_manager.load_index(index_path, doc_map_path)
                print("Loaded FAISS index vectors:", embedding_manager.index.ntotal)
                print("Loaded doc_map type:", type(embedding_manager.doc_map), "length:", len(embedding_manager.doc_map))
                assert embedding_manager.index.ntotal == len(embedding_manager.doc_map), "FAISS index and doc_map are misaligned!"
            except Exception as e:
                print(f"Error loading index for {preset_name}: {e}")
        else:
            print(f"No index found for {preset_name}, skipping load.")

    def rename_preset(self, old_name):
        import os
        presets = self.app_instance.teleprompt_config.setdefault("presets", {})
        if old_name not in presets:
            print(f"Preset '{old_name}' not found; cannot rename.")
            return

        # Build UI for renaming
        rename_layout = BoxLayout(orientation='vertical', spacing=10, padding=10)
        instructions = Label(
            text=f"Enter new name for '{old_name}':",
            size_hint=(1, None),
            height=40,
            font_size='20sp'
        )
        rename_input = TextInput(
            text=old_name,
            multiline=False,
            size_hint=(1, None),
            height=50,
            font_size='20sp'
        )
        rename_layout.add_widget(instructions)
        rename_layout.add_widget(rename_input)

        btn_bar = BoxLayout(orientation='horizontal', spacing=10, size_hint=(1, None), height=60)
        confirm_btn = Button(text="Confirm", font_size='20sp')
        cancel_btn = Button(text="Cancel", font_size='20sp')
        btn_bar.add_widget(confirm_btn)
        btn_bar.add_widget(cancel_btn)
        rename_layout.add_widget(btn_bar)

        rename_popup = Popup(
            title="Rename Preset",
            content=rename_layout,
            size_hint=(0.7, 0.4),
            auto_dismiss=False
        )
        rename_popup.open()

        def do_rename(_btn):
            new_name = rename_input.text.strip()
            if not new_name:
                print("No new name entered; skipping rename.")
                rename_popup.dismiss()
                return
            if new_name in presets:
                print(f"Preset '{new_name}' already exists; cannot rename.")
                rename_popup.dismiss()
                return

            # 1) Move config data from old_name to new_name (all associated documents remain)
            presets[new_name] = presets.pop(old_name)

            # 2) Rename the folder on disk so that your FAISS index and doc_map remain linked.
            old_folder = os.path.join("presets", old_name)
            new_folder = os.path.join("presets", new_name)
            if os.path.exists(old_folder):
                try:
                    os.rename(old_folder, new_folder)
                    print(f"Renamed folder from '{old_folder}' to '{new_folder}'")
                except Exception as e:
                    print(f"Error renaming folder: {e}")
            else:
                try:
                    os.makedirs(new_folder, exist_ok=True)
                    print(f"Created folder '{new_folder}' for preset '{new_name}'")
                except Exception as e:
                    print(f"Error creating folder '{new_folder}': {e}")

            # 3) If this preset was active, update the active preset variable
            if self.app_instance.selected_preset == old_name:
                self.app_instance.selected_preset = new_name

            # 4) Save settings to update the YAML config file
            self.app_instance.save_settings()

            rename_popup.dismiss()

            # 5) Refresh the UI by rebuilding the preset list and updating any displayed labels
            self.rebuild_preset_list()
            self.update_active_preset_label()
            self.update_preset_button_highlight()

            # 6) Optionally, update the current preset popup title (if open)
            if hasattr(self, 'current_preset_popup') and self.current_preset_popup:
                self.current_preset_popup.title = new_name

        confirm_btn.bind(on_press=do_rename)
        cancel_btn.bind(on_press=lambda x: rename_popup.dismiss())

    def delete_preset(self, old_name):
        import shutil
        confirm_layout = BoxLayout(orientation='vertical', spacing=10, padding=10)
        info_lbl = Label(
            text=f"Are you sure you want to DELETE '{old_name}'?\nThis removes docs & FAISS index.",
            size_hint=(1, None),
            height=60,
            font_size='18sp',
            halign='center',
            valign='middle'
        )
        info_lbl.bind(size=lambda lbl, val: setattr(lbl, 'text_size', val))
        confirm_layout.add_widget(info_lbl)

        btn_bar = BoxLayout(orientation='horizontal', spacing=10, size_hint=(1, None), height=50)
        yes_btn = Button(text="Yes, Delete", font_size='18sp')
        no_btn = Button(text="No, Cancel", font_size='18sp')
        btn_bar.add_widget(yes_btn)
        btn_bar.add_widget(no_btn)
        confirm_layout.add_widget(btn_bar)

        confirm_popup = Popup(
            title="Confirm Delete",
            content=confirm_layout,
            size_hint=(0.7, 0.4),
            auto_dismiss=False
        )

        def do_delete(_btn):
            confirm_popup.dismiss()
            presets = self.app_instance.teleprompt_config.setdefault("presets", {})
            if old_name not in presets:
                print(f"Preset '{old_name}' not found; cannot delete.")
                return

            # 1) Remove the old preset block
            presets.pop(old_name)
            if self.app_instance.selected_preset == old_name:
                self.app_instance.selected_preset = None

            # Save the updated config immediately so the old preset is removed from YAML
            self.app_instance.save_settings()

            # 2) Create a new empty preset entry (e.g. "Preset 1")
            reset_name = "Preset 1"
            presets[reset_name] = {"parsed_documents": []}

            # Ensure the new preset folder exists on disk
            new_folder = os.path.join("presets", reset_name)
            if not os.path.exists(new_folder):
                os.makedirs(new_folder, exist_ok=True)

            # Save again so that teleprompt_config.yaml reflects the new preset entry
            self.app_instance.save_settings()
            print(f"Preset '{old_name}' deleted and reset to '{reset_name}'")

            # 3) Refresh the UI so that the preset buttons update
            self.rebuild_preset_list()
            self.update_active_preset_label()
            self.update_preset_button_highlight()

            old_folder = os.path.join("presets", old_name)
            if os.path.exists(old_folder):
                try:
                    shutil.rmtree(old_folder)
                    print(f"Deleted folder: {old_folder}")
                except Exception as e:
                    print(f"Error deleting folder '{old_folder}': {e}")

        yes_btn.bind(on_press=do_delete)
        no_btn.bind(on_press=lambda x: confirm_popup.dismiss())
        confirm_popup.open()

    def update_preset_button_highlight(self):
        active = self.app_instance.selected_preset
        for name, btn in self.preset_buttons.items():
            # Clear previous instructions
            btn.canvas.before.clear()
            # Reset default background
            btn.background_color = (0.6, 0.6, 0.6, 1)
            btn.color = (0, 0, 0, 1)

            if name == active:
                # Draw green border
                with btn.canvas.before:
                    Color(0, 1, 0, 1)
                    Line(width=3, rectangle=(btn.x, btn.y, btn.width, btn.height))

    def _redraw_border(self, *args):
        self.update_preset_button_highlight()

    def update_active_preset_label(self):
        self.active_preset_label.text = self._get_active_preset_text()

    def _get_active_preset_text(self):
        if self.app_instance.selected_preset:
            return f"{self.app_instance.selected_preset}"
        else:
            return "No Preset Active"

    def go_back(self, instance):
        if self.manager:
            self.manager.current = "main"
        else:
            print("No ScreenManager found; cannot go back.")

    def open_file_manager_for_preset(self, preset_name):
        import tkinter as tk
        from tkinter import filedialog
        import os

        # Create and hide the Tkinter root window
        root = tk.Tk()
        root.withdraw()  # Hide the main tkinter window

        # Set the initial directory to the preset folder if it exists
        initial_dir = os.path.abspath(f"presets/{preset_name}")
        if not os.path.exists(initial_dir):
            initial_dir = os.getcwd()

        # Open the native file selection dialog
        file_path = filedialog.askopenfilename(
            initialdir=initial_dir,
            title="Select a file to upload",
            filetypes=[("All Files", "*.*")]
        )
        root.destroy()  # Destroy the Tkinter window

        if file_path:
            # Process the selected file
            self.process_file_for_preset(preset_name, file_path)
        else:
            print("No file selected.")

    def process_file_for_preset(self, preset_name, file_path):
        import os
        try:
            # --- Ensure the preset folder exists ---
            preset_folder = os.path.join("presets", preset_name)
            if not os.path.exists(preset_folder):
                os.makedirs(preset_folder, exist_ok=True)
                print(f"Created directory for preset: {preset_folder}")

            # Start the upload timer...
            self.layout.add_widget(self.timer_label)
            self.start_upload_timer()

            # 1) Perform your OCR/embedding operation
            from teleprompt.main import parse_file  # Adjust the import if necessary
            ocr_engine_choice = self.app_instance.teleprompt_config.get('ocr_engine', "Tesseract (Fast)")
            parsed_data = parse_file(file_path, ocr_engine_choice, None)
            text_content = parsed_data.get("content", "")

            # 2) Update preset config with file info
            presets = self.app_instance.teleprompt_config.setdefault("presets", {})
            if preset_name not in presets:
                presets[preset_name] = {"parsed_documents": []}
            current_preset = presets[preset_name]
            if "parsed_documents" not in current_preset:
                current_preset["parsed_documents"] = []
            relative_file_path = os.path.relpath(file_path, start=os.getcwd())
            preset_data = {
                "content": text_content,
                "file_type": os.path.splitext(file_path)[1].lstrip('.'),
                "filename": relative_file_path
            }
            current_preset["parsed_documents"].append(preset_data)
            self.app_instance.teleprompt_config[preset_name] = current_preset
            self.app_instance.save_settings()

            # 3) Define paths
            index_path = os.path.join(preset_folder, "index.faiss")
            doc_map_path = os.path.join(preset_folder, "doc_map.npy")
            print(f"Index path: {index_path}")
            print(f"Doc map path: {doc_map_path}")

            # 4) Incrementally update the FAISS index using the shared instance
            if os.path.exists(index_path) and os.path.exists(doc_map_path):
                embedding_manager.load_index(index_path, doc_map_path)
            else:
                # If no existing index, build a new one with the new document as the only data.
                embedding_manager.build_index([(os.path.basename(file_path), text_content)],
                                                index_path, doc_map_path)
            # Append new document data.
            new_doc = [(os.path.basename(file_path), text_content)]
            embedding_manager.add_documents(new_doc, index_path, doc_map_path)
 
            # Trigger asynchronous update with the new document.
            new_docs = [(os.path.basename(file_path), text_content)]
            preset_folder = os.path.join("presets", preset_name)
            index_path = os.path.join(preset_folder, "index.faiss")
            doc_map_path = os.path.join(preset_folder, "doc_map.npy")
            embedding_manager.async_update_index(new_docs, index_path, doc_map_path,
                 on_update_done=lambda: print("Index updated after document upload."))

            # 5) Stop the timer and remove it from the layout
            self.stop_upload_timer()
            if self.timer_label in self.layout.children:
                self.layout.remove_widget(self.timer_label)

            print(f"File '{file_path}' uploaded and processed for preset '{preset_name}'.")
        except Exception as e:
            print(f"Error processing file: {e}")
            from kivy.uix.popup import Popup
            from kivy.uix.label import Label
            error_popup = Popup(
                title="Error",
                content=Label(text=str(e)),
                size_hint=(0.6, 0.3)
            )
            error_popup.open()
            self.stop_upload_timer()
            if self.timer_label in self.layout.children:
                self.layout.remove_widget(self.timer_label)

    def update_settings_display(self):
        """
        Updates the UI display for the current preset's documents.
        For example, if you have a Label (self.documents_display) showing uploaded docs,
        update its text to list the base filenames of the documents.
        Adjust this example to match your actual UI.
        """
        import os
        # Fetch the presets from teleprompt_config
        presets = self.app_instance.teleprompt_config.get("presets", {})
        # Get the currently active preset's data
        current_preset = presets.get(self.app_instance.selected_preset, {})
        documents = current_preset.get("parsed_documents", [])
        
        # If you have a widget to display document names, update it.
        if hasattr(self, 'documents_display'):
            if documents:
                self.documents_display.text = "Preset Documents:\n" + "\n".join(
                    os.path.basename(doc["filename"]) for doc in documents
                )
            else:
                self.documents_display.text = "Preset Documents"
        else:
            # Otherwise, simply print the list to the console.
            if documents:
                print("Preset Documents:\n" + "\n".join(os.path.basename(doc["filename"]) for doc in documents))
            else:
                print("Preset Documents")

    def start_timer(self):
        """Starts the stopwatch timer."""
        self.start_time = time.time()
        # Schedule update_timer to run every 0.01 seconds (10ms)
        self.timer_event = Clock.schedule_interval(self.update_timer, 0.01)

    def update_timer(self, dt):
        """Updates the timer label with elapsed time (MM:SS:ms)."""
        elapsed = time.time() - self.start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        milliseconds = int((elapsed - int(elapsed)) * 1000)
        self.timer_label.text = f"{minutes:02d}:{seconds:02d}:{milliseconds:03d}"

    def stop_timer(self):
        """Stops the stopwatch timer."""
        if hasattr(self, 'timer_event'):
            self.timer_event.cancel()

    def start_upload_timer(self):
        """Starts the upload stopwatch timer."""
        self.start_time = time.time()
        # Schedule an update every 0.05s (20 times per second)
        self.timer_event = Clock.schedule_interval(self._update_upload_timer, 0.05)

    def _update_upload_timer(self, dt):
        """Updates the timer label with elapsed time (MM:SS:ms)."""
        elapsed = time.time() - self.start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        milliseconds = int((elapsed - int(elapsed)) * 1000)
        self.timer_label.text = f"{minutes:02d}:{seconds:02d}:{milliseconds:03d}"

    def stop_upload_timer(self):
        """Stops the upload timer and resets the timer label."""
        if hasattr(self, 'timer_event'):
            self.timer_event.cancel()
            del self.timer_event
        self.timer_label.text = "00:00:000"

    def rebuild_preset_list(self):
        """
        Clears the middle_layout and rebuilds it based on self.app_instance.teleprompt_config["presets"].
        """
        # Clear the old preset rows and reset the buttons dictionary
        self.middle_layout.clear_widgets()
        self.preset_buttons = {}
        
        # Retrieve all presets from config
        presets_dict = self.app_instance.teleprompt_config.get("presets", {})
        
        for preset_name in presets_dict:
            # Create a row for each preset
            row = BoxLayout(orientation='horizontal', spacing=20, size_hint=(1, 1))
            
            # Create the preset button (text set exactly as in config)
            preset_btn = Button(
                text=preset_name,
                size_hint=(1, .6),
                font_size='45sp',
                background_normal='',
                background_color=(0.6, 0.6, 0.6, 1),
                color=(0, 0, 0, 1),
                bold=True
            )
            # Bind the button to open its popup (use a lambda to capture the current preset_name)
            preset_btn.bind(on_press=lambda inst, nm=preset_name: self.open_preset_popup(nm))
            preset_btn.bind(pos=self._redraw_border, size=self._redraw_border)
            
            # Save reference for later (for example, in renaming)
            self.preset_buttons[preset_name] = preset_btn
            
            # Center the button using left/right spacers
            row.add_widget(Widget(size_hint_x=0.25))
            row.add_widget(preset_btn)
            row.add_widget(Widget(size_hint_x=0.25))
            
            self.middle_layout.add_widget(row)
        
        # Optionally update the active preset highlight
        self.update_preset_button_highlight()
        self.update_active_preset_label()
        self.update_preset_button_highlight()