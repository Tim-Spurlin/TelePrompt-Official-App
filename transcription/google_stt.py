from google.cloud import speech
import numpy as np

class GoogleSpeechToText:
    def __init__(self, sample_rate):
        self.client = speech.SpeechClient()
        self.config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=int(sample_rate),
            language_code="en-US",
            model="phone_call",  # Use "phone_call" for telephone audio
            use_enhanced=True,
        )
        # Enable interim results for streaming
        self.streaming_config = speech.StreamingRecognitionConfig(
            config=self.config,
            interim_results=True  # Changed from False to True
        )

    def transcribe(self, audio_segment):
        """Transcribes a single audio segment."""
        # Convert to int16 *FIRST*:
        audio_segment_int16 = (audio_segment * 32767).astype(np.int16)
        audio_bytes = audio_segment_int16.tobytes()
        requests = [speech.StreamingRecognizeRequest(audio_content=audio_bytes)]

        try:
            responses = self.client.streaming_recognize(self.streaming_config, requests)
            for response in responses:
                if response.results:
                    for result in response.results:
                        if result.alternatives:
                            transcript_text = result.alternatives[0].transcript
                            return transcript_text
            return None  # No transcript found
        except Exception as e:
            print(f"Transcription error: {e}")
            return None

    def stream_transcribe(self, audio_generator):
        """
        Streams transcription of audio chunks and yields interim/final transcripts.
        
        Args:
            audio_generator: Generator yielding audio chunks
            
        Yields:
            Tuple of (status, transcript) where status is one of:
            - "interim": Partial transcription
            - "final": Complete transcription
            - "error": Error message
        """
        requests = (speech.StreamingRecognizeRequest(audio_content=chunk)
                   for chunk in audio_generator)

        try:
            responses = self.client.streaming_recognize(self.streaming_config, requests)

            for response in responses:
                if not response.results:
                    continue

                result = response.results[0]
                if not result.alternatives:
                    continue

                transcript = result.alternatives[0].transcript
                
                if result.is_final:
                    yield "final", transcript
                else:
                    yield "interim", transcript

        except Exception as e:
            print(f"Streaming transcription error: {e}")
            yield "error", str(e)