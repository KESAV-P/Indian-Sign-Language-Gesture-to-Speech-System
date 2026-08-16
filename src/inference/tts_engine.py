"""
Text-to-Speech (TTS) Engine for ISL-Speak.
Wraps offline pyttsx3 as primary engine, with gTTS online fallback and safe headless output logging.
"""

import os
import threading
import sys

class TTSEngine:
    """
    Offline/Online Text-to-Speech Engine.
    """

    def __init__(self, rate: int = 150, volume: float = 1.0, voice_gender: str = "female"):
        self.rate = rate
        self.volume = volume
        self.voice_gender = voice_gender
        self.pyttsx_available = False
        self.engine = None

        # Attempt initializing pyttsx3 (offline primary)
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", self.rate)
            self.engine.setProperty("volume", self.volume)

            # Try selecting voice gender
            voices = self.engine.getProperty("voices")
            if voices:
                for voice in voices:
                    if voice_gender.lower() in voice.name.lower():
                        self.engine.setProperty("voice", voice.id)
                        break

            self.pyttsx_available = True
        except Exception as e:
            print(f"[TTSEngine Warning] pyttsx3 init failed ({e}). Fallback modes enabled.")

    def speak(self, text: str, async_mode: bool = True):
        """
        Converts text to speech output.
        """
        if not text or not text.strip():
            return

        clean_text = text.strip()
        print(f"🗣️ [TTS Speaking]: '{clean_text}'")

        if async_mode:
            thread = threading.Thread(target=self._speak_sync, args=(clean_text,))
            thread.daemon = True
            thread.start()
        else:
            self._speak_sync(clean_text)

    def _speak_sync(self, text: str):
        if self.pyttsx_available and self.engine:
            try:
                # pyttsx3 is thread-sensitive on mac; handle carefully
                self.engine.say(text)
                self.engine.runAndWait()
                return
            except Exception as e:
                print(f"[TTSEngine Warning] pyttsx3 runtime error ({e}). Trying gTTS...")

        # Fallback to gTTS if online
        try:
            from gtts import gTTS
            import tempfile
            import os

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
            # Final fallback: text output
            print(f"[TTSEngine Silent Fallback]: '{text}'")


if __name__ == "__main__":
    tts = TTSEngine()
    tts.speak("Hello, welcome to Indian Sign Language Gesture to Speech System.", async_mode=False)
