"""
Text-to-Speech (TTS) Engine for ISL-Speak.
Wraps offline pyttsx3 as primary engine, with gTTS online fallback and safe headless output logging.
Uses a single persistent worker thread and queue for thread safety.
"""

import os
import sys
import queue
import threading

class TTSEngine:
    """
    Offline/Online Text-to-Speech Engine using a persistent worker thread and queue.
    """

    def __init__(self, rate: int = 150, volume: float = 1.0, voice_gender: str = "female"):
        self.rate = rate
        self.volume = volume
        self.voice_gender = voice_gender
        self.pyttsx_available = False
        self.engine = None
        self.lock = threading.Lock()
        self.speech_queue = queue.Queue()

        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def _init_pyttsx3(self):
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)
            voices = engine.getProperty("voices")
            if voices:
                for voice in voices:
                    if self.voice_gender.lower() in voice.name.lower():
                        engine.setProperty("voice", voice.id)
                        break
            return engine
        except Exception as e:
            print(f"[TTSEngine Warning] pyttsx3 init failed ({e}). Fallback modes enabled.")
            return None

    def _worker_loop(self):
        self.engine = self._init_pyttsx3()
        if self.engine:
            self.pyttsx_available = True

        while True:
            text = self.speech_queue.get()
            if text is None:
                self.speech_queue.task_done()
                break
            try:
                self._speak_sync(text)
            finally:
                self.speech_queue.task_done()

    def speak(self, text: str, async_mode: bool = True):
        """
        Converts text to speech output.
        """
        if not text or not text.strip():
            return

        clean_text = text.strip()
        print(f"🗣️ [TTS Speaking]: '{clean_text}'")

        if async_mode:
            self.speech_queue.put(clean_text)
        else:
            self.speech_queue.put(clean_text)
            self.speech_queue.join()

    def _speak_sync(self, text: str):
        with self.lock:
            if self.pyttsx_available and self.engine:
                try:
                    self.engine.say(text)
                    self.engine.runAndWait()
                    return
                except Exception as e:
                    print(f"[TTSEngine Warning] pyttsx3 runtime error ({e}). Trying gTTS...")

            # Fallback to gTTS if online
            try:
                from gtts import gTTS
                import tempfile

                tts = gTTS(text=text, lang="en", slow=False)
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
                    temp_filename = fp.name
                
                tts.save(temp_filename)

                # Play mp3 based on platform
                if sys.platform == "darwin":
                    os.system(f"afplay '{temp_filename}' >/dev/null 2>&1")
                elif sys.platform.startswith("linux"):
                    os.system(f"mpg123 '{temp_filename}' >/dev/null 2>&1")
                elif sys.platform == "win32":
                    os.system(f"start /min mplay32 /play /close '{temp_filename}'")
                
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
            except Exception as fallback_err:
                print(f"[TTSEngine Silent Fallback]: '{text}'")

    def join(self):
        """Wait for all queued speech tasks to finish."""
        self.speech_queue.join()


if __name__ == "__main__":
    tts = TTSEngine()
    tts.speak("Hello, welcome to Indian Sign Language Gesture to Speech System.", async_mode=False)
