import sounddevice as sd
import numpy as np
import logging
from utils.file_utils import load_config  # Ensure this function is correctly implemented to load the settings.

logging.basicConfig(level=logging.WARNING)

def list_audio_devices():
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        print(f"[{idx}]: {dev['name']} (Input: {dev['max_input_channels']} channels, Output: {dev['max_output_channels']} channels)")

def find_playback_capture_device():
    """
    Finds and returns the user-selected audio device from the settings file.
    If no device is selected, defaults to B1 if available.
    """
    config = load_config("config/teleprompt_config.yaml") or {}  # Load settings
    device_index = config.get('device_index', None)  # Get selected device index
    
    # If device_index is None, try to auto-select a B1 device
    if device_index is None:
        devices = sd.query_devices()
        for idx, dev in enumerate(devices):
            if "b1" in dev["name"].lower():
                device_index = idx
                print(f"Auto-selected B1 virtual device: {dev['name']} at index {idx}")
                break
    
    # Default to system-selected input if no B1 was found
    if device_index is None:
        print("Warning: No B1 device found. Using system default input.")

    sample_rate = 48000  # Standard sample rate
    device_info = sd.query_devices(device_index, kind='input')
    device_name = device_info['name']

    return sample_rate, device_name, device_index  # Return sample rate, name, and index