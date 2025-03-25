# audio_processing/vad.py

import numpy as np

class VoiceActivityDetector:
    def __init__(self, sample_rate, rms_multiplier=2.5, pause_seconds=0.75, lookbehind_seconds=0.2, buffer=None):
        self.sample_rate = sample_rate
        self.rms_multiplier = rms_multiplier
        self.pause_seconds = pause_seconds
        self.lookbehind_seconds = lookbehind_seconds
        self.lookbehind_samples = int(lookbehind_seconds * sample_rate)
        self.min_recording_seconds = 0.5
        self.block_size = int(0.1 * sample_rate)  # 100ms per block
        self.below_threshold_counter = 0
        self.rms_threshold = 0.0
        self.speech_started = False
        self.recording = False
        self.recording_start_index = 0
        self.segment_ready = None
        self.buffer = buffer

    def calibrate(self, calibration_data):
        self.rms_threshold = np.mean(self._calculate_rms(calibration_data)) * self.rms_multiplier
        print(f"Calibration complete. RMS threshold: {self.rms_threshold}")

    def _calculate_rms(self, audio_chunk):
        return np.sqrt(np.mean(np.square(audio_chunk)))

    def process_chunk(self, chunk):
        rms = self._calculate_rms(chunk)
        if not self.speech_started:
            if rms > self.rms_threshold:
                print("VAD DETECTED SPEECH START")
                self.speech_started = True
                self.recording_start_index = (
                    self.buffer.buffer_index - len(chunk) - self.lookbehind_samples
                ) % self.buffer.buffer_size
                if self.recording_start_index < 0:
                    self.recording_start_index = 0
                self.recording = True
                self.below_threshold_counter = 0
                return True  # Speech start
        elif self.recording:
            if rms < self.rms_threshold:
                self.below_threshold_counter += 1
            else:
                self.below_threshold_counter = 0
            if self.below_threshold_counter * self.block_size / self.sample_rate >= self.pause_seconds:
                print("VAD DETECTED PAUSE")
                self.recording = False
                self.speech_started = False
                self.segment_ready = True  # Signal segment ready
                return True
        return False

    def get_segment(self):
        if self.segment_ready:
            self.segment_ready = None
            return True
        return None


class CircularAudioBuffer:
    def __init__(self, buffer_seconds, lookbehind_seconds, sample_rate):
        self.sample_rate = sample_rate
        self.buffer_size = int(buffer_seconds * sample_rate)
        self.lookbehind_samples = int(lookbehind_seconds * sample_rate)
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.buffer_index = 0

    def add_data(self, chunk):
        chunk_size = len(chunk)
        if self.buffer_index + chunk_size <= self.buffer_size:
            self.buffer[self.buffer_index : self.buffer_index + chunk_size] = chunk
            self.buffer_index += chunk_size
        else:
            remaining = self.buffer_size - self.buffer_index
            self.buffer[self.buffer_index :] = chunk[:remaining]
            self.buffer[: chunk_size - remaining] = chunk[remaining:]
            self.buffer_index = chunk_size - remaining

    def retrieve_segment(self, vad_instance):
        end_index = self.buffer_index
        start_index = vad_instance.recording_start_index
        if start_index < end_index:
            return self.buffer[start_index:end_index]
        else:
            return np.concatenate((self.buffer[start_index:], self.buffer[:end_index]))

    def get_lookbehind(self):
        start_index = (self.buffer_index - self.lookbehind_samples) % self.buffer_size
        if start_index < self.buffer_index:
            return self.buffer[start_index : self.buffer_index]
        else:
            return np.concatenate((self.buffer[start_index:], self.buffer[:self.buffer_index]))

    def clear(self):
        self.buffer[:] = 0.0
        self.buffer_index = 0