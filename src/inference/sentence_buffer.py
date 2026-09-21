"""
Sentence Buffer & Anti-Flicker Filter for Real-time Gesture Predictions.
Filters frame-level classification noise using majority voting over a rolling window K,
suppresses consecutive duplicate words, and builds structured sentence text for TTS output.
"""

from collections import deque, Counter
from typing import List, Optional


class SentenceBuffer:
    """
    Anti-flicker buffer that accumulates predictions, applies majority voting,
    prevents immediate word repetitions, and flushes sentences to speech.

    Tuning guide:
      window_size      – number of model predictions to accumulate (lower = faster response).
      min_frequency    – how many of those predictions must agree (< window_size = more forgiving).
      repeat_cooldown  – how many *accepted* words must pass before the same word fires again
                         (set to 0 to allow immediate repeats, 1 blocks back-to-back only).
    """

    def __init__(
        self,
        window_size: int = 3,
        min_confidence: float = 0.50,
        min_frequency: int = 2,
        repeat_cooldown: int = 1,
        max_words: int = 15,
    ):
        self.window_size = window_size
        self.min_confidence = min_confidence
        # min_frequency: minimum votes inside window_size to accept a word
        self.min_frequency = min(min_frequency, window_size)
        self.repeat_cooldown = repeat_cooldown
        self.max_words = max_words
        self.prediction_window: deque = deque(maxlen=window_size)
        self.accepted_words: List[str] = []
        # Ring buffer of recently accepted words for repeat suppression
        self._recent: deque = deque(maxlen=max(repeat_cooldown, 1))

    def add_prediction(self, word: str, confidence: float) -> Optional[str]:
        """
        Adds a single window prediction to the rolling buffer.

        Returns:
            Optional[str]: Newly accepted word if majority consensus reached, else None.
        """
        if confidence < self.min_confidence or word.lower() == "background":
            return None

        self.prediction_window.append(word)

        # Only evaluate once we have a full window
        if len(self.prediction_window) < self.window_size:
            return None

        counts = Counter(self.prediction_window)
        most_common_word, frequency = counts.most_common(1)[0]

        if frequency >= self.min_frequency:
            # Suppress immediate repeats based on cooldown
            if most_common_word not in self._recent:
                self.accepted_words.append(most_common_word)
                self._recent.append(most_common_word)
                # Slide the window forward (don't clear — allows overlapping detections)
                # Pop oldest entry so buffer isn't stuck
                for _ in range(self.min_frequency):
                    if self.prediction_window:
                        self.prediction_window.popleft()

                if len(self.accepted_words) >= self.max_words:
                    self.flush()

                return most_common_word

        return None

    def get_current_sentence(self) -> str:
        """Returns currently assembled sentence string."""
        return " ".join(self.accepted_words)

    def flush(self) -> str:
        """
        Flushes and clears accepted words, returning the complete sentence.
        """
        sentence = self.get_current_sentence()
        self.accepted_words.clear()
        self._recent.clear()
        self.prediction_window.clear()
        return sentence

    def clear(self):
        """Clears all accumulated state."""
        self.accepted_words.clear()
        self._recent.clear()
        self.prediction_window.clear()


if __name__ == "__main__":
    buf = SentenceBuffer(window_size=3, min_confidence=0.50, min_frequency=2, repeat_cooldown=1)

    # Simulate a stream of noisy frame predictions
    stream = [
        ("Hello", 0.85),
        ("Hello", 0.90),
        ("Hi", 0.40),   # Below threshold — ignored
        ("Hello", 0.88), # window = [Hello, Hello] → 2/3 → accept Hello
        ("Thank You", 0.85),
        ("Thank You", 0.89),
        ("Thank You", 0.95), # window = [Thank You, Thank You] → 2/3 → accept Thank You
    ]

    for word, conf in stream:
        accepted = buf.add_prediction(word, conf)
        if accepted:
            print(f"Accepted new word: '{accepted}'")

    sentence = buf.get_current_sentence()
    print(f"Assembled Sentence: '{sentence}'")
    assert "Hello" in sentence and "Thank You" in sentence, f"Got: '{sentence}'"
    print("Self-test passed.")
