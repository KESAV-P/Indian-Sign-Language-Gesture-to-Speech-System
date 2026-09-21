package com.isl.speak.ml

import com.isl.speak.network.NetworkClient
import com.isl.speak.network.PredictRequest
import com.isl.speak.network.PredictResponse
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.ArrayDeque

/**
 * Manages landmark sequence buffering (45 frames x 258 features) and backend communication.
 * Clears the buffer on prediction to enforce non-overlapping inference windows (matching Python logic).
 */
class LandmarkBufferManager(
    val sequenceLength: Int = 45
) {
    private val frameBuffer = ArrayDeque<FloatArray>()

    @Synchronized
    fun addFrame(vector: FloatArray) {
        if (frameBuffer.size >= sequenceLength) {
            frameBuffer.removeFirst()
        }
        frameBuffer.addLast(vector)
    }

    @Synchronized
    fun isBufferFull(): Boolean {
        return frameBuffer.size >= sequenceLength
    }

    @Synchronized
    fun getSequencePayload(): List<List<Float>> {
        return frameBuffer.map { floatArray -> floatArray.toList() }
    }

    @Synchronized
    fun clear() {
        frameBuffer.clear()
    }

    /**
     * Sends buffered 45x258 sequence to backend, clears buffer, and returns PredictResponse or null on error.
     */
    suspend fun submitPrediction(baseUrl: String, modelType: String = "lstm"): PredictResponse? {
        if (!isBufferFull()) return null

        val payload = getSequencePayload()
        // Clear buffer immediately to enforce non-overlapping windows
        clear()

        return withContext(Dispatchers.IO) {
            try {
                val service = NetworkClient.getService(baseUrl)
                val response = service.predictGesture(PredictRequest(sequence = payload, modelType = modelType))
                if (response.isSuccessful) {
                    response.body()
                } else {
                    null
                }
            } catch (e: Exception) {
                e.printStackTrace()
                null
            }
        }
    }
}
