package com.isl.speak.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.core.RunningMode

/**
 * Wraps MediaPipe PoseLandmarker + HandLandmarker Tasks SDK.
 *
 * For each CameraX ImageProxy frame, runs both landmarkers and assembles
 * the 258-dim FloatArray that EXACTLY matches the Python training format:
 *   - Indices 0..131   : Pose (33 × [x, y, z, visibility])
 *   - Indices 132..194 : Left Hand (21 × [x, y, z])
 *   - Indices 195..257 : Right Hand (21 × [x, y, z])
 */
class ISLGestureAnalyzer(
    context: Context,
    private val onVectorReady: (FloatArray) -> Unit
) {
    companion object {
        private const val TAG = "ISLGestureAnalyzer"
        private const val POSE_MODEL = "pose_landmarker_lite.task"
        private const val HAND_MODEL = "hand_landmarker.task"
    }

    private var poseLandmarker: PoseLandmarker? = null
    private var handLandmarker: HandLandmarker? = null

    init {
        setupPoseLandmarker(context)
        setupHandLandmarker(context)
    }

    private fun setupPoseLandmarker(context: Context) {
        try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath(POSE_MODEL)
                .setDelegate(Delegate.CPU)
                .build()
            val options = PoseLandmarker.PoseLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.IMAGE)
                .setNumPoses(1)
                .build()
            poseLandmarker = PoseLandmarker.createFromOptions(context, options)
        } catch (e: Exception) {
            Log.e(TAG, "PoseLandmarker init failed: ${e.message}")
        }
    }

    private fun setupHandLandmarker(context: Context) {
        try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath(HAND_MODEL)
                .setDelegate(Delegate.CPU)
                .build()
            val options = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.IMAGE)
                .setNumHands(2)
                .build()
            handLandmarker = HandLandmarker.createFromOptions(context, options)
        } catch (e: Exception) {
            Log.e(TAG, "HandLandmarker init failed: ${e.message}")
        }
    }

    /**
     * Processes a CameraX ImageProxy. Converts to Bitmap, runs landmarkers,
     * assembles the 258-dim vector, and delivers it via onVectorReady.
     * MUST be called from a background thread; closes imageProxy when done.
     */
    fun analyzeFrame(imageProxy: ImageProxy) {
        val bitmap = imageProxy.toBitmap().rotate(imageProxy.imageInfo.rotationDegrees.toFloat())

        val mpImage = BitmapImageBuilder(bitmap).build()

        // --- Pose ---
        var poseLandmarkPoints: List<LandmarkVectorExtractor.LandmarkPoint>? = null
        poseLandmarker?.let { pl ->
            val poseResult = pl.detect(mpImage)
            if (poseResult.landmarks().isNotEmpty()) {
                poseLandmarkPoints = poseResult.landmarks()[0].map { lm ->
                    LandmarkVectorExtractor.LandmarkPoint(
                        x = lm.x(),
                        y = lm.y(),
                        z = lm.z(),
                        visibility = lm.visibility().orElse(0.0f)
                    )
                }
            }
        }

        // --- Hands ---
        var leftHandPoints: List<LandmarkVectorExtractor.LandmarkPoint>? = null
        var rightHandPoints: List<LandmarkVectorExtractor.LandmarkPoint>? = null

        handLandmarker?.let { hl ->
            val handResult = hl.detect(mpImage)
            val landmarksList = handResult.landmarks()
            val handednessList = handResult.handedness()

            for (i in landmarksList.indices) {
                // MediaPipe handedness "Left"/"Right" from the model's perspective
                // (front camera = mirrored, so we swap to match Python's Holistic convention)
                val label = handednessList.getOrNull(i)?.firstOrNull()?.categoryName() ?: continue
                val points = landmarksList[i].map { lm ->
                    LandmarkVectorExtractor.LandmarkPoint(x = lm.x(), y = lm.y(), z = lm.z())
                }
                // Python Holistic labels hands from the subject's perspective (not the camera's).
                // MediaPipe Tasks HandLandmarker on a FRONT camera image returns the model's view,
                // which is mirrored — so "Left" in Tasks == subject's Right hand, and vice versa.
                when (label) {
                    "Left"  -> rightHandPoints = points   // model's Left = subject's Right
                    "Right" -> leftHandPoints  = points   // model's Right = subject's Left
                }
            }
        }

        val vector = LandmarkVectorExtractor.extractFrameVector(
            poseLandmarkPoints,
            leftHandPoints,
            rightHandPoints
        )

        imageProxy.close()
        onVectorReady(vector)
    }

    fun close() {
        poseLandmarker?.close()
        handLandmarker?.close()
    }

    // ---------- helpers ----------

    private fun Bitmap.rotate(degrees: Float): Bitmap {
        if (degrees == 0f) return this
        val matrix = Matrix().apply { postRotate(degrees) }
        return Bitmap.createBitmap(this, 0, 0, width, height, matrix, true)
    }
}
