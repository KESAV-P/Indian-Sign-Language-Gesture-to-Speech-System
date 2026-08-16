"""
Sentence Buffer & Anti-Flicker Filter for Real-time Gesture Predictions.
Filters frame-level classification noise using majority voting over a rolling window K,
suppresses consecutive duplicate words, and builds structured sentence text for TTS output.
"""

from collections import deque, Counter
from typing import List, Optional, Tuple


class SentenceBuffer:
    """
    Anti-flicker buffer that accumulates predictions, applies majority voting,
    prevents immediate word repetitions, and flushes sentences to speech.
    """

    def __init__(self, window_size: int = 5, min_confidence: float = 0.6, max_words: int = 15):
        self.window_size = window_size
        self.min_confidence = min_confidence
        self.max_words = max_words
        self.prediction_window = deque(maxlen=window_size)
        self.accepted_words: List[str] = []
        self.last_accepted_word: Optional[str] = None

    def add_prediction(self, word: str, confidence: float) -> Optional[str]:
        """
        Adds a single window prediction to the rolling buffer.
        
        Returns:
            Optional[str]: Newly accepted word if majority consensus reached, else None.
        """
        if confidence < self.min_confidence or word.lower() == "background":
            return None

        self.prediction_window.append(word)

        # Check if buffer has reached full window size
        if len(self.prediction_window) == self.window_size:
            counts = Counter(self.prediction_window)
            most_common_word, frequency = counts.most_common(1)[0]

            # Require majority consensus (e.g. 4/5 or 5/5 agreement)
            if frequency >= (self.window_size // 2 + 1):
                if most_common_word != self.last_accepted_word:
                    self.accepted_words.append(most_common_word)
                    self.last_accepted_word = most_common_word
                    self.prediction_window.clear()

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
        self.last_accepted_word = None
        self.prediction_window.clear()
        return sentence

    def clear(self):
        """Clears all accumulated state."""
        self.accepted_words.clear()
        self.last_accepted_word = None
        self.prediction_window.clear()


if __name__ == "__main__":
    buf = SentenceBuffer(window_size=5, min_confidence=0.6)
    
    # Simulate a stream of noisy frame predictions: ["hello", "hello", "hi", "hello", "hello"]
    stream = [
        ("hello", 0.85),
        ("hello", 0.90),
        ("hi", 0.40), # Below threshold
        ("hello", 0.88),
        ("hello", 0.92), # Should trigger majority acceptance of "hello"
        ("thank_you", 0.85),
        ("thank_you", 0.89),
        ("thank_you", 0.95),
        ("thank_you", 0.91), # Should accept "thank_you"
    ]

    for word, conf in stream:
        accepted = buf.add_prediction(word, conf)
        if accepted:
            print(f"Accepted new word: '{accepted}'")

    sentence = buf.flush()
    print(f"Final Assembled Sentence: '{sentence}'")
    assert sentence == "hello thank_you", f"Expected 'hello thank_you', got '{sentence}'"
