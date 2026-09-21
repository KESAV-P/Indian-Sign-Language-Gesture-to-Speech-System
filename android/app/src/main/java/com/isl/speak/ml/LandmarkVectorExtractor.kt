package com.isl.speak.ml

/**
 * Assembles 258-dimensional landmark feature vectors per frame.
 * Layout MUST match src/preprocessing/extract_landmarks.py in Python:
 * - Indices 0..131   (132 floats): Pose (33 points * 4 [x, y, z, visibility])
 * - Indices 132..194 (63 floats) : Left Hand (21 points * 3 [x, y, z])
 * - Indices 195..257 (63 floats) : Right Hand (21 points * 3 [x, y, z])
 * Total: 258 float features.
 */
object LandmarkVectorExtractor {

    const val POSE_FEATURES = 132       // 33 * 4
    const val LEFT_HAND_FEATURES = 63   // 21 * 3
    const val RIGHT_HAND_FEATURES = 63  // 21 * 3
    const val TOTAL_FEATURES = 258      // 132 + 63 + 63

    data class LandmarkPoint(
        val x: Float,
        val y: Float,
        val z: Float,
        val visibility: Float = 0.0f
    )

    /**
     * Constructs a 258-dim FloatArray from pose, left hand, and right hand landmark lists.
     */
    fun extractFrameVector(
        poseLandmarks: List<LandmarkPoint>?,
        leftHandLandmarks: List<LandmarkPoint>?,
        rightHandLandmarks: List<LandmarkPoint>?
    ): FloatArray {
        val vector = FloatArray(TOTAL_FEATURES)

        // 1. Pose (Indices 0..131)
        if (poseLandmarks != null && poseLandmarks.size >= 33) {
            for (i in 0 until 33) {
                val lm = poseLandmarks[i]
                val baseIdx = i * 4
                vector[baseIdx] = lm.x
                vector[baseIdx + 1] = lm.y
                vector[baseIdx + 2] = lm.z
                vector[baseIdx + 3] = lm.visibility
            }
        }

        // 2. Left Hand (Indices 132..194)
        if (leftHandLandmarks != null && leftHandLandmarks.size >= 21) {
            for (i in 0 until 21) {
                val lm = leftHandLandmarks[i]
                val baseIdx = POSE_FEATURES + (i * 3)
                vector[baseIdx] = lm.x
                vector[baseIdx + 1] = lm.y
                vector[baseIdx + 2] = lm.z
            }
        }

        // 3. Right Hand (Indices 195..257)
        if (rightHandLandmarks != null && rightHandLandmarks.size >= 21) {
            for (i in 0 until 21) {
                val lm = rightHandLandmarks[i]
                val baseIdx = POSE_FEATURES + LEFT_HAND_FEATURES + (i * 3)
                vector[baseIdx] = lm.x
                vector[baseIdx + 1] = lm.y
                vector[baseIdx + 2] = lm.z
            }
        }

        return vector
    }
}
