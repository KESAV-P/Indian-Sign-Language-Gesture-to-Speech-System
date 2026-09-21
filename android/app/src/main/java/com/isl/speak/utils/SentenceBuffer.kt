package com.isl.speak.utils

import java.util.ArrayDeque

/**
 * Kotlin port of Anti-Flicker SentenceBuffer.
 * Filters gesture prediction noise via majority voting over a rolling window,
 * prevents rapid duplicates, and builds translated sentence strings.
 */
class SentenceBuffer(
    val windowSize: Int = 3,
    val minConfidence: Float = 0.45f,
    val minFrequency: Int = 2,
    val repeatCooldown: Int = 1,
    val maxWords: Int = 15
) {
    private val predictionWindow = ArrayDeque<String>()
    private val acceptedWords = mutableListOf<String>()
    private val recentAccepted = ArrayDeque<String>()

    /**
     * Adds a prediction to the buffer.
     * Returns newly accepted word if majority consensus reached, else null.
     */
    @Synchronized
    fun addPrediction(word: String, confidence: Float): String? {
        if (confidence < minConfidence || word.equals("background", ignoreCase = true)) {
            return null
        }

        if (predictionWindow.size >= windowSize) {
            predictionWindow.removeFirst()
        }
        predictionWindow.addLast(word)

        if (predictionWindow.size < windowSize) {
            return null
        }

        // Count frequency of words in prediction window
        val frequencyMap = HashMap<String, Int>()
        for (w in predictionWindow) {
            frequencyMap[w] = (frequencyMap[w] ?: 0) + 1
        }

        var mostCommonWord: String? = null
        var maxFreq = 0
        for ((w, freq) in frequencyMap) {
            if (freq > maxFreq) {
                maxFreq = freq
                mostCommonWord = w
            }
        }

        if (mostCommonWord != null && maxFreq >= minFrequency) {
            if (!recentAccepted.contains(mostCommonWord)) {
                acceptedWords.add(mostCommonWord)
                if (recentAccepted.size >= repeatCooldown) {
                    recentAccepted.removeFirst()
                }
                recentAccepted.addLast(mostCommonWord)

                // Slide prediction window
                repeat(minOf(minFrequency, predictionWindow.size)) {
                    if (!predictionWindow.isEmpty()) {
                        predictionWindow.removeFirst()
                    }
                }

                if (acceptedWords.size >= maxWords) {
                    flush()
                }

                return mostCommonWord
            }
        }

        return null
    }

    @Synchronized
    fun getCurrentSentence(): String {
        return acceptedWords.joinToString(" ")
    }

    @Synchronized
    fun flush(): String {
        val sentence = getCurrentSentence()
        acceptedWords.clear()
        recentAccepted.clear()
        predictionWindow.clear()
        return sentence
    }

    @Synchronized
    fun clear() {
        acceptedWords.clear()
        recentAccepted.clear()
        predictionWindow.clear()
    }
}
