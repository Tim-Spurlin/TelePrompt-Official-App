import sys
from kivy.core.window import Window

if sys.platform.startswith('win'):
    import ctypes
    from ctypes import wintypes
    SPI_GETWORKAREA = 48
    rcWorkArea = wintypes.RECT()
    ctypes.windll.user32.SystemParametersInfoW(SPI_GETWORKAREA, 0, ctypes.byref(rcWorkArea), 0)
    work_width  = rcWorkArea.right - rcWorkArea.left
    work_height = rcWorkArea.bottom - rcWorkArea.top

from kivy.config import Config

# Force the window to use a fixed size and custom position from the very start.
Config.set('graphics', 'position', 'custom')
Config.set('graphics', 'left', '100')    # Position ~100px from left edge (adjust as needed)
Config.set('graphics', 'top', '50')       # Position ~50px from top edge (adjust as needed)
Config.set('graphics', 'width', '800')    # Fixed width
Config.set('graphics', 'height', '780')   # Fixed safe height (1080 - 300)

# Now import Kivy and the rest of your modules
import kivy  # <---- Import kivy FIRST
# Suppress debug logs for libraries that are being too verbose.
import logging
logging.basicConfig(level=logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("kivy").setLevel(logging.WARNING)

kivy.require('2.3.1')  # Replace with your actual Kivy version

# --- NEW: Set Kivy Logger Level to WARNING or ERROR ---
from kivy.logger import Logger, LOG_LEVELS
Logger.setLevel(LOG_LEVELS["warning"]) # Or Logger.setLevel(LOG_LEVELS["error"]) for even less output
import sys
import os
import time
import requests
import threading  # Import threading
import sounddevice as sd
import numpy as np
from google.cloud import speech
import queue  # <--- Import queue here
import wave
from pystray import Icon as TrayIcon, Menu as TrayMenu, MenuItem as TrayMenuItem
from PIL import Image

# Kivy/KivyMD imports
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
from kivy.uix.filechooser import FileChooserIconView  
from kivy.uix.popup import Popup
from kivy.uix.slider import Slider
from kivy.properties import StringProperty, NumericProperty, ObjectProperty, BooleanProperty
from kivy.clock import Clock
from kivymd.app import MDApp
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.scrollview import ScrollView  # Import ScrollView

# Project module imports
from audio_processing import audio_capture
from audio_processing.vad import VoiceActivityDetector, CircularAudioBuffer
from transcription.google_stt import GoogleSpeechToText  # Correct import
import ai_response.ai_interface as ai_interface
import ai_response.base_prompt_handler as base_prompt_handler
from utils.file_utils import load_config, save_config
from ui.settings_screen import SettingsScreen  # <--- Import Settings Screen
from semantic_search import embed_and_compare
from embedding_manager import embedding_manager
from ai_response.base_prompt_handler import create_prompt
from ai_response.ai_interface import get_ai_response, load_api_key_from_env

# Add project root to path (for module imports)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# --- Configuration ---
SAMPLE_RATE = 48000
CHANNELS = 1
DTYPE = np.float32
BLOCK_SIZE = int(0.1 * SAMPLE_RATE)
LOOKBEHIND_SECONDS = 0.2
LOOKBEHIND_SAMPLES = int(LOOKBEHIND_SECONDS * SAMPLE_RATE)
BUFFER_SECONDS = 5
BUFFER_SIZE = int(BUFFER_SECONDS * SAMPLE_RATE)
CALIBRATION_SECONDS = 2
RMS_MULTIPLIER = 2.5
PAUSE_SECONDS = 0.75
MIN_RECORDING_SECONDS = 0.5
import os
import ctypes
import logging

logger = logging.getLogger(__name__)

def load_voicemeeter_settings(xml_file_path="teleprompt_voicemeeter_settings.xml"):
    """Loads Voicemeeter settings from an XML file using VoicemeeterRemote64.dll."""
    try:
        # Load VoicemeeterRemote64.dll (assuming it's in the same directory as your exe)
        VoicemeeterRemote64 = ctypes.WinDLL("VoicemeeterRemote64.dll")

        # Define the function signature for VMR_LoadSettingsW (Unicode version)
        VMR_LoadSettingsW = VoicemeeterRemote64.VMR_LoadSettingsW
        VMR_LoadSettingsW.argtypes = [ctypes.w_wchar_p]  # Wide char path

        # Ensure the XML file path is absolute for robustness in compiled app
        absolute_xml_path = os.path.abspath(xml_file_path)

        # Call VMR_LoadSettingsW with the XML file path (Unicode)
        result = VMR_LoadSettingsW(absolute_xml_path)

        if result == 1:  # VBAN_API_OK
            logger.info(f"Voicemeeter settings loaded successfully from: {xml_file_path}")
            return True
        else:
            logger.error(f"Error loading Voicemeeter settings (Code: {result}) from: {xml_file_path}")
            return False

    except Exception as e:
        logger.error(f"Error loading Voicemeeter settings: {e}")
        return False
    
import asyncio
class TelePromptApp(MDApp):
    selected_device = StringProperty("Default")  # Available for SettingsScreen

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load Voicemeeter settings on startup
        settings_loaded = load_voicemeeter_settings()
        if not settings_loaded:
            print("Warning: Voicemeeter settings could not be loaded. Please ensure Voicemeeter is installed and running.")
        else:
            print("Voicemeeter settings loaded successfully.")
        
        self.full_conversation_memory = [] # Store full conversation history
        
        # Use the shared EmbeddingManager instance
        self.embedding_manager = embedding_manager

        # Load your configuration once
        self.teleprompt_config = load_config("config/teleprompt_config.yaml") or {}

        # Set default VAD settings BEFORE usage:
        self.rms_multiplier = 2.5
        self.pause_seconds = 0.75
        self.lookbehind_seconds = 0.2

        # Determine the selected preset from config, if valid
        self.selected_preset = self.teleprompt_config.get("last_preset")
        if not self.selected_preset or self.selected_preset not in self.teleprompt_config.get("presets", {}):
            self.selected_preset = None

        # If a valid preset exists, load its Faiss index (if files exist)
        if self.selected_preset:
            asyncio.ensure_future(self.ensure_index_ready())

        # Initialize remaining properties
        self.recording_count = 0
        self.current_segment = None
        self.current_preset_context = ""
        self.is_running = False  # Track processing state

        # Tesseract OCR setup (if needed)
        self.is_gpu_ocr_enabled = False
        self.ocr = None

        # Setup audio buffer, VAD, and transcription queue
        self.audio_callback_called = False
        self.testing_mode = False
        self.audio_buffer = CircularAudioBuffer(BUFFER_SECONDS, LOOKBEHIND_SECONDS, SAMPLE_RATE)
        self.transcription_queue = queue.Queue()
        self.vad = VoiceActivityDetector(SAMPLE_RATE, RMS_MULTIPLIER, PAUSE_SECONDS, LOOKBEHIND_SECONDS, buffer=self.audio_buffer)
        self.transcriber = None
        self.stream = None
        self.file_manager = None

        # Load shared calibration settings, if available
        shared = self.teleprompt_config.get("shared_calibration", None)
        if shared:
            self.vad.rms_threshold = shared.get("rms_threshold", None)
            self.lookbehind_seconds = shared.get("lookbehind_seconds", self.lookbehind_seconds)
            self.pause_seconds = shared.get("pause_seconds", self.pause_seconds)
            self.calibration_status = "Calibrated"
        else:
            self.calibration_status = "Not yet calibrated"

        # Define responses_layout available for UI updates
        self.responses_layout = BoxLayout(orientation='vertical', size_hint_y=None, spacing=5)
        self.responses_layout.bind(minimum_height=self.responses_layout.setter('height'))

        # Create the Start/Stop button here so it's available in build()
        self.start_stop_button = Button(
            text="Start",
            size_hint=(1, 1),
        )
        self.start_stop_button.bind(on_press=self.toggle_start_stop)

        # Define scroll_view so it's available in build()
        self.scroll_view = ScrollView()
        
        # Create an inner content layout for scroll_view (if needed)
        scroll_content = BoxLayout(orientation='vertical', size_hint_y=None)
        scroll_content.bind(minimum_height=scroll_content.setter('height'))
        self.scroll_view.add_widget(scroll_content)


        # IMPORTANT: Add responses_layout to scroll_content
        scroll_content.add_widget(self.responses_layout)

    def build(self):
        self.status_label = Label(
            text=f"Preset: {self.selected_preset}" if self.selected_preset else "No Preset Selected",
            size_hint=(1, None),
            height=40,
            halign='center',
            font_size=46,
            bold=True,
            color=(1, 1, 1, 1)
        )

        # Create a Settings button
        settings_button = Button(
            text="Settings",
            size_hint=(1, 1),
        )
        # Bind the settings button event to open_settings method.
        settings_button.bind(on_press=self.open_settings)

        # Create a bottom bar layout for the Start/Stop and Settings buttons
        bottom_bar = BoxLayout(orientation='horizontal', spacing=10, size_hint=(1, None), height=60)
        bottom_bar.add_widget(self.start_stop_button)
        bottom_bar.add_widget(settings_button)

        main_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        main_layout.add_widget(self.status_label)
        main_layout.add_widget(self.scroll_view)
        main_layout.add_widget(bottom_bar)  # Add the bottom bar to the main layout

        # Wrap the main layout in a Screen widget for the ScreenManager.
        main_screen = Screen(name="main")
        main_screen.add_widget(main_layout)

        self.screen_manager = ScreenManager()
        self.screen_manager.add_widget(main_screen)
        
        # Add the settings screen so that switching to "settings" works.
        from ui.settings_screen import SettingsScreen  # adjust the import path if needed
        settings_screen = SettingsScreen(config=self.teleprompt_config, name="settings", app_instance=self)
        self.screen_manager.add_widget(settings_screen)
        
        
        return self.screen_manager
        
    def update_start_button_state(self):
        """
        Checks if the current preset has at least one document
        (by loading the doc_map). If documents exist the Start button
        is enabled; otherwise it is disabled and a message is shown.
        """
        current_preset = self.selected_preset
        if not current_preset:
            self.start_stop_button.disabled = True
            return


    def update_label_size(self, *args):
      pass
        # self.response_display.text_size = (Window.width - 20, None) #Removed
        # self.response_display.texture_update()

    def open_settings(self, instance):
        self.screen_manager.current = "settings"

    def go_to_main_screen(self):
        self.screen_manager.current = "main"

    def toggle_start_stop(self, instance):
        """
        Called when the user presses the Start/Stop button.
        Only start processing if the button is enabled.
        """
        if self.start_stop_button.disabled:
            # Should not be called while disabled, but just in case:
            popup = Popup(
                title="No Documents",
                content=Label(text="You must upload documents for this preset first."),
                size_hint=(None, None),
                size=(400, 200),
                auto_dismiss=True
            )
            popup.open()
            return

        # Normal Start/Stop logic:
        if self.is_running:
            self.stop_processing()
            instance.text = "Start"
        else:
            self.start_processing()
            instance.text = "Stop"

    def start_processing(self):

        preset_folder = os.path.join("presets", self.selected_preset)
        index_path = os.path.join(preset_folder, "index.faiss")
        if not os.path.exists(index_path):
            popup = Popup(
                title="No Documents",
                content=Label(text="Please upload documents for this preset before starting."),
                size_hint=(None, None),
                size=(400, 200),
                auto_dismiss=True
            )
            popup.open()
            return

        # Proceed with start processing as before...
        if self.is_running:
            return

        self.is_running = True
        print("Starting audio processing...")
        try:
            # Retrieve the correct device from settings
            sample_rate, device_name, device_index = audio_capture.find_playback_capture_device()

            self.audio_buffer = CircularAudioBuffer(BUFFER_SECONDS, LOOKBEHIND_SECONDS, sample_rate)
            self.vad = VoiceActivityDetector(sample_rate, self.rms_multiplier, self.pause_seconds, self.lookbehind_seconds, buffer=self.audio_buffer)

            self.transcriber = GoogleSpeechToText(sample_rate)

            # Ensure selected device index is correctly passed to InputStream
            self.stream = sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=BLOCK_SIZE,
                callback=self.audio_callback,
                device=device_index  # Use user-selected device
            )
            self.stream.start()
            self.start_streaming_transcription_thread()
            print(f"Audio stream started on device: {device_name} (index: {device_index})")

        except Exception as e:
            print(f"Error starting processing: {e}")

    def stop_processing(self):
        self.is_running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        if hasattr(self, 'transcription_thread_stop_event') and self.transcription_thread and self.transcription_thread.is_alive():
            print("Stopping transcription thread...")
            self.transcription_queue.put(None)  # Send a sentinel value to stop thread
            self.transcription_thread_stop_event.set()  # Signal the thread to stop
            self.transcription_thread.join(timeout=2)  # Wait up to 2 seconds for thread termination
            if self.transcription_thread.is_alive():
                print("Transcription thread did not terminate gracefully.")
            self.transcription_thread = None

    def audio_callback(self, indata, frames, time_info, status):
        if self.testing_mode:
            # In test mode, only set the flag and do nothing else.
            self.audio_callback_called = True
            return
        self.audio_callback_called = True
        if status:
            print(f"Sounddevice Status: {status}")

        chunk = indata[:, 0].copy()
        self.audio_buffer.add_data(chunk)
        
        # Accumulate audio data while recording
        if self.vad.recording:
            # --- NEW: Put audio chunk into transcription queue ---
            self.transcription_queue.put(chunk.copy())  # <--- Add this line
            # --- End of new code ---
            if self.current_segment is None:
                self.current_segment = chunk.copy()
            else:
                self.current_segment = np.concatenate((self.current_segment, chunk))
        
        if self.vad.process_chunk(chunk):
            if self.vad.get_segment() is not None:
                audio_segment = self.audio_buffer.retrieve_segment(self.vad)
                # Optional: self.save_recording(audio_segment)
                self.current_segment = None
                # --- REMOVED: Old batch processing call ---

    def get_device_list(self):
        devices = sd.query_devices()
        return [(i, device['name']) for i, device in enumerate(devices) if device['max_input_channels'] > 0]

    def save_settings(self, *args):
        """Saves the current configuration to the config file."""
        save_config(self.teleprompt_config, "config/teleprompt_config.yaml")

    def select_path(self, path):
      '''Called when a file/directory is selected.'''
      print(f"Selected path: {path}") #debug
      # Store the selected file path in your configuration, associated with the current preset.
      current_preset = self.teleprompt_config.get("presets", {}).get(self.selected_preset, {})
      if 'documents' not in current_preset:
          current_preset['documents'] = []
      current_preset['documents'].append(path)
      self.teleprompt_config["presets"][self.selected_preset] = current_preset
      self.save_settings() #Save the updates
      self.update_settings_display() # refresh display

    def on_rms_change(self, instance, value):
      self.rms_multiplier = value
      instance.parent.children[1].text = f"RMS Multiplier: {value}"
      # Update the config
      self.teleprompt_config.setdefault('presets', {}).setdefault(self.selected_preset, {})['rms_multiplier'] = value
      self.save_settings() # Save immediately

    def on_pause_change(self, instance, value):
        self.pause_seconds = value
        instance.parent.children[1].text = f"Pause Seconds: {value}"
        self.teleprompt_config.setdefault('presets', {}).setdefault(self.selected_preset, {})['pause_seconds'] = value
        self.save_settings()

    def on_lookbehind_change(self, instance, value):
        self.lookbehind_seconds = value
        instance.parent.children[1].text = f"Lookbehind Seconds: {value}"
        self.teleprompt_config.setdefault('presets', {}).setdefault(self.selected_preset, {})['lookbehind_seconds'] = value
        self.save_settings()

    def on_stop(self):
        # Ensure processing is stopped when the app closes
        self.stop_processing()
        self.save_settings()  # Save settings on close

    def start_streaming_transcription_thread(self):
        """Starts the streaming transcription thread."""
        if not self.is_running or not self.transcriber:
            return  # Do not start if not running or transcriber unavailable

        self.transcription_thread_stop_event = threading.Event()  # Event to control thread stop
        self.transcription_thread = threading.Thread(target=self.streaming_transcription_loop, daemon=True)
        self.transcription_thread.start()
        print("Streaming transcription thread started.")

    def streaming_transcription_loop(self):
        """
        Continuously processes audio chunks from the transcription queue,
        sends them to Google STT for streaming transcription, and handles responses.
        """
        try:
            while not self.transcription_thread_stop_event.is_set() and self.is_running:
                try:
                    audio_chunk = self.transcription_queue.get(timeout=0.1)  # Non-blocking get with timeout
                except queue.Empty:
                    continue  # No audio chunk available, loop again

                if audio_chunk is None:  # Poison pill to stop the thread
                    print("Transcription thread received stop signal.")
                    break

                # --- MODIFIED audio chunk generator (yields continuously) ---
                def audio_chunk_generator():
                    # Continue yielding until VAD is no longer recording
                    while True:
                        try:
                            # Wait for a chunk with a timeout of 0.1 seconds
                            chunk = self.transcription_queue.get(timeout=0.1)
                        except queue.Empty:
                            if self.vad.recording:
                                # Yield a silent chunk if VAD is active to maintain the stream
                                silent = np.zeros(self.audio_buffer.buffer_size // 10, dtype=np.float32)  # Adjust chunk length as needed
                                yield (silent * 32767).astype(np.int16).tobytes()
                                continue
                            else:
                                break
                        if chunk is None:  # Sentinel value to stop the generator
                            break
                        yield (chunk * 32767).astype(np.int16).tobytes()
                # --- End of modified generator ---

                # Process the stream of transcription responses
                for result_type, transcript in self.transcriber.stream_transcribe(audio_chunk_generator()):
                    if result_type == "interim":
                        print(f"Interim transcript: {transcript}") # Go away for now
                    elif result_type == "final":
                        print(f"Final transcript for AI processing: {transcript}") # Debug: Final transcript being processed
                        Clock.schedule_once(lambda dt: self.process_transcript_for_ai(transcript), 0) # Schedule UI update on main thread
                    elif result_type == "error":
                        print(f"Streaming error: {transcript}")
        except Exception as e:
            print(f"Error in streaming_transcription_loop: {e}") # General error in loop
            return  # Exit loop on critical error

    def process_transcript_for_ai(self, transcript_caller):
        """
        Processes a final transcript (from streaming) for AI response generation,
        storing persistent conversation history (with role information) and using it to build the prompt.
        """
        if not transcript_caller:
            print("No transcript received from Google STT.")
            return

        print(f"Interviewer says (for AI): {transcript_caller}")
        
        preset_folder = os.path.join("presets", self.selected_preset)
        
        # Save the caller's transcript with who="Caller" into persistent history.
        self.embedding_manager.add_conversation_utterance(transcript_caller, preset_folder, who="Caller")
        
        # Append the caller's message to the full conversation memory
        self.full_conversation_memory.append({
            "who": "Caller",
            "content": transcript_caller
        })
        
            
        faiss_convo = self.embedding_manager.query_conversation_history(
            query_text=transcript_caller, 
            preset_folder=preset_folder, 
            top_k=5
        )
        
        if not faiss_convo:
            faiss_convo = self.embedding_manager.recent_history_per_preset.get(preset_folder, [])
            
        combined_conversation = list(self.full_conversation_memory) + list(faiss_convo)
        
        # Retrieve conversation history (as a list of dictionaries).
        conversation_history = self.embedding_manager.query_conversation_history(transcript_caller, preset_folder, top_k=5)
        # Fallback to in-memory recent history if query returns nothing.
        if not conversation_history:
            conversation_history = self.embedding_manager.recent_history_per_preset.get(preset_folder, [])

        # Query document index for additional context.
        top_results = self.embedding_manager.query(transcript_caller, top_k=5)
        doc_texts_list = []
        for doc_id, chunk_text, score in top_results:
            doc_texts_list.append(f"[DocID: {doc_id} Score: {score}]\n{chunk_text}")

        # Build the prompt using the persistent conversation history.
        prompt_for_ai = base_prompt_handler.create_prompt(conversation_history=combined_conversation, caller_transcription_buffer=transcript_caller, user_documents=doc_texts_list)

        print("Sending prompt to AI...")
        api_key_gemini = ai_interface.load_api_key_from_env()
        ai_response_text = ai_interface.get_ai_response(prompt_for_ai, api_key=api_key_gemini)
        print("AI response received.")
        
        # Save the AI response with who="Assistant" in persistent history.
        self.embedding_manager.add_conversation_utterance(ai_response_text, preset_folder, who="Assistant")
        self.full_conversation_memory.append({
            "who": "Assistant",
            "content": ai_response_text
        })

        # Update the UI with the AI response.
        self.add_response_label(ai_response_text)

    def add_response_label(self, response_text):
        response_label = Label(
            text=f"{response_text}",
            size_hint_y=None,
            halign='center',
            font_size=46,
            bold=True,
            color=(1, 1, 1, 1)  # Default white for the newest
        )
        response_label.bind(width=lambda instance, value: setattr(instance, 'text_size', (value, None)))
        response_label.bind(texture_size=lambda instance, value: setattr(instance, 'height', value[1]))
        
        # Insert the new label at the top so that it's the newest
        self.responses_layout.add_widget(response_label, index=0)
        # Update all response label colors
        self.update_response_colors()
        print("Added response label to layout")
        Clock.schedule_once(lambda dt: self.scroll_view.scroll_to(response_label, padding=10, animate=True), 0)

    def update_response_colors(self):
        # Iterate over each child to update their colors.
        for i, child in enumerate(self.responses_layout.children):
            if i == 0:
                # Newest: bright white
                child.color = (1, 1, 1, 1)
            elif i == 1:
                # Second newest: slightly faded gray
                child.color = (0.7, 0.7, 0.7, 1)
            else:
                # Older responses: lighter gray
                child.color = (0.4, 0.4, 0.4, 1)

    async def ensure_index_ready(self):
        """
        Asynchronously ensure that the FAISS index and doc_map are ready.
        If they exist, load them; otherwise, trigger an asynchronous background update.
        """
        preset_folder = os.path.join("presets", self.selected_preset)
        index_path = os.path.join(preset_folder, "index.faiss")
        doc_map_path = os.path.join(preset_folder, "doc_map.npy")
        try:
            if os.path.exists(index_path) and os.path.exists(doc_map_path):
                self.embedding_manager.load_index(index_path, doc_map_path)
                print("FAISS index loaded successfully.")
                print("Loaded FAISS index vectors:", self.embedding_manager.index.ntotal)
                print("Loaded doc_map type:", type(self.embedding_manager.doc_map), "length:", len(self.embedding_manager.doc_map))
                assert self.embedding_manager.index.ntotal == len(self.embedding_manager.doc_map), "FAISS index and doc_map are misaligned!"
            else:
                documents = self.get_documents_for_preset(self.selected_preset)
                if documents:
                    self.build_index_async(preset_folder, index_path, doc_map_path)
                else:
                    print("No documents available to build the FAISS index.")
        except Exception as e:
            print(f"Error in ensure_index_ready: {e}")

    def on_preset_selected(self, preset_name):
        """
        Handles a new preset selection by saving settings and triggering index readiness.
        """
        self.selected_preset = preset_name
        self.teleprompt_config["last_preset"] = preset_name
        self.save_settings()
        # Trigger the asynchronous index load/update
        asyncio.ensure_future(self.ensure_index_ready())

    def build_index_async(self, preset_folder, index_path, doc_map_path):
        """
        Start a background thread to build the FAISS index.
        """
        print("Starting background thread to build FAISS index...")
        thread = threading.Thread(
            target=self.build_index_for_preset,
            args=(preset_folder, index_path, doc_map_path),
            daemon=True
        )
        thread.start()

    def ensure_index_ready(self):
        """
        Ensures that the FAISS index and doc_map for the selected preset are available.
        If they exist, attempts to load them; otherwise, builds the index from available documents.
        If building is required, it is done asynchronously to prevent UI blocking.
        """
        if not self.selected_preset:
            print("No preset selected; cannot ensure index readiness.")
            return

        preset_folder = os.path.join("presets", self.selected_preset)
        index_path = os.path.join(preset_folder, "index.faiss")
        doc_map_path = os.path.join(preset_folder, "doc_map.npy")
        
        if os.path.exists(index_path) and os.path.exists(doc_map_path):
            try:
                self.embedding_manager.load_index(index_path, doc_map_path)
                print("FAISS index loaded successfully.")
            except Exception as e:
                print(f"Error loading index: {e}")
                # If loading fails, rebuild in the background.
                self.build_index_async(preset_folder, index_path, doc_map_path)
        else:
            # If files do not exist, check if documents are available
            documents = self.get_documents_for_preset(self.selected_preset)
            if documents:
                # Build the index asynchronously
                self.build_index_async(preset_folder, index_path, doc_map_path)
            else:
                print("No documents available to build the FAISS index.")

    def build_index_for_preset(self, preset_folder, index_path, doc_map_path):
        """
        Builds (and saves) the FAISS index for the current preset.
        Documents are expected to be a list of tuples with (doc_id, text_content).
        This method is called in a background thread by build_index_async.
        """
        documents = self.get_documents_for_preset(self.selected_preset)
        if documents:
            try:
                self.embedding_manager.build_index(documents, index_path, doc_map_path)
                print("FAISS index built successfully in background.")
            except Exception as e:
                print(f"Error building FAISS index: {e}")
        else:
            print("No documents found to build an index.")

    def get_documents_for_preset(self, preset_name):
        """
        Retrieves the documents for a preset from the teleprompt_config.
        Expects documents to be stored under 'parsed_documents' in the preset data.
        Returns a list of (doc_id, text_content) tuples.
        """
        presets = self.teleprompt_config.get("presets", {})
        preset_data = presets.get(preset_name, {})
        # Adjust key names as per your configuration structure.
        return [(i, doc.get("content", "")) for i, doc in enumerate(preset_data.get("parsed_documents", []))]

    def reload_updated_index(self, index_path, doc_map_path):
        self.embedding_manager.load_index(index_path, doc_map_path)
        print("Updated FAISS index reloaded into memory.")

def parse_file(file_path, ocr_engine_choice, ocr):
    """
    Reads a file from disk and returns a dictionary with:
      - "filename": the file path,
      - "file_type": e.g., 'pdf', 'docx', 'txt', or 'image',
      - "content": the extracted text.
    Uses Tesseract only.
    """
    print(f"parse_file called with path: {file_path}, engine: {ocr_engine_choice}, ocr: {ocr}")
    file_lower = file_path.lower()

    if file_lower.endswith(".pdf"):
        from ocr_handler import extract_text_from_pdf_tesseract
        content = extract_text_from_pdf_tesseract(file_path)
        print(f"PDF content extracted (first 100 chars): {content[:100]}...")
        return {"filename": file_path, "file_type": "pdf", "content": content}
    elif file_lower.endswith(".docx"):
        from ocr_handler import extract_text_from_docx
        content = extract_text_from_docx(file_path)
        print(f"DOCX content extracted (first 100 chars): {content[:100]}...")
        return {"filename": file_path, "file_type": "docx", "content": content}
    elif file_lower.endswith(".doc"):
        return {"filename": file_path, "file_type": "doc", "content": "Unable to parse .doc files directly. Convert to .docx first."}
    elif file_lower.endswith(".txt"):
        from ocr_handler import extract_text_from_txt
        content = extract_text_from_txt(file_path)
        print(f"TXT content extracted (first 100 chars): {content[:100]}...")
        return {"filename": file_path, "file_type": "txt", "content": content}
    else:
        # Assume image file.
        from ocr_handler import extract_text
        # Pass Tesseract as the engine (if needed) or adjust extraction accordingly.
        content = extract_text(file_path, ocr_engine_choice, None)
        print(f"Image content extracted (first 100 chars): {content[:100]}...")
        return {"filename": file_path, "file_type": "image", "content": content}

def setup_system_tray():
    # Convert your PNG/ICO to a PIL Image
    icon_image = Image.open("my_tray_icon.png")

    # Build a menu for the tray icon (optional)
    tray_menu = TrayMenu(
        TrayMenuItem("Show Window", on_show_window),
        TrayMenuItem("Exit", on_exit_app)
    )

    tray_icon = TrayIcon("My App", icon_image, "My App - System Tray", menu=tray_menu)
    tray_icon.run()

def on_show_window(icon, item):
    # For example, show your Kivy window if it’s hidden, or do something else
    pass

def on_exit_app(icon, item):
    # Cleanly exit your Kivy app
    icon.stop()

if __name__ == "__main__":
    # Start tray in a separate thread so it doesn’t block Kivy
    tray_thread = threading.Thread(target=setup_system_tray, daemon=True)
    tray_thread.start()

    # Now start your Kivy app
    TelePromptApp().run()

# Example usage:
query = "Your query text here"
docs = [
    "Document text for file one",
    "Another document text",
    "More document content..."
]

best_match = embed_and_compare(query, docs)
print("Best matched document:", best_match)