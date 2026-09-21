package com.isl.speak

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.isl.speak.ml.ISLGestureAnalyzer
import com.isl.speak.ml.LandmarkBufferManager
import com.isl.speak.tts.AndroidTTSEngine
import com.isl.speak.utils.SentenceBuffer
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {

    private var ttsEngine: AndroidTTSEngine? = null
    private val sentenceBuffer = SentenceBuffer()
    private val bufferManager = LandmarkBufferManager()
    private var gestureAnalyzer: ISLGestureAnalyzer? = null

    private val permissionState = mutableStateOf(false)
    private val currentStatus = mutableStateOf("Warming Up...")
    private val currentWord = mutableStateOf("Point camera at ISL gesture")
    private val currentConfidence = mutableStateOf(0.0f)
    private val currentSentence = mutableStateOf("")
    private val serverUrl = mutableStateOf("http://10.0.2.2:8000") // emulator default; change to LAN IP for real device

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        permissionState.value = isGranted
        if (isGranted) initGestureAnalyzer()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        ttsEngine = AndroidTTSEngine(this)
        checkCameraPermission()

        setContent {
            ISLSpeakTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = Color(0xFF0D1117)
                ) {
                    if (permissionState.value) {
                        MainCameraScreen(
                            status = currentStatus.value,
                            word = currentWord.value,
                            confidence = currentConfidence.value,
                            sentence = currentSentence.value,
                            serverUrl = serverUrl.value,
                            onServerUrlChange = { serverUrl.value = it },
                            onFrameReady = { imageProxy ->
                                gestureAnalyzer?.analyzeFrame(imageProxy)
                                    ?: imageProxy.close()
                            }
                        )
                    } else {
                        PermissionDeniedScreen(
                            onRequestPermission = { checkCameraPermission() }
                        )
                    }
                }
            }
        }
    }

    private fun checkCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            permissionState.value = true
            initGestureAnalyzer()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun initGestureAnalyzer() {
        gestureAnalyzer = ISLGestureAnalyzer(this) { vector ->
            onNewFrameVector(vector)
        }
    }

    private fun onNewFrameVector(vector: FloatArray) {
        bufferManager.addFrame(vector)

        val filled = bufferManager.getSequencePayload().size
        if (bufferManager.isBufferFull()) {
            currentStatus.value = "Sending to backend..."
            CoroutineScope(Dispatchers.Main).launch {
                val response = bufferManager.submitPrediction(serverUrl.value)
                if (response != null) {
                    currentStatus.value = response.status.replace("_", " ").uppercase()
                    currentWord.value = response.label
                    currentConfidence.value = response.confidence

                    val accepted = sentenceBuffer.addPrediction(response.label, response.confidence)
                    if (accepted != null) {
                        ttsEngine?.speak(accepted)
                    }
                    currentSentence.value = sentenceBuffer.getCurrentSentence()
                } else {
                    currentStatus.value = "SERVER OFFLINE"
                    currentWord.value = "Check backend URL"
                    currentConfidence.value = 0.0f
                }
            }
        } else {
            currentStatus.value = "Collecting (${filled * 100 / 45}%)"
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        ttsEngine?.shutdown()
        gestureAnalyzer?.close()
    }
}

@Composable
fun MainCameraScreen(
    status: String,
    word: String,
    confidence: Float,
    sentence: String,
    serverUrl: String,
    onServerUrlChange: (String) -> Unit,
    onFrameReady: (androidx.camera.core.ImageProxy) -> Unit
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current   // fixed: uses lifecycle-runtime-compose

    val isOffline = status.contains("OFFLINE", ignoreCase = true)
    val statusColor = when {
        isOffline -> Color(0xFFD32F2F)
        status.contains("TRANSLATED", ignoreCase = true) -> Color(0xFF2E7D32)
        else -> Color(0xFF1565C0)
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFF0D1117))
            .padding(horizontal = 16.dp, vertical = 12.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // ── Header ──────────────────────────────────────────────────────────
        Text(
            text = "ISL Speak",
            fontSize = 26.sp,
            fontWeight = FontWeight.Bold,
            color = Color.White
        )
        Text(
            text = "Indian Sign Language → Speech",
            fontSize = 13.sp,
            color = Color(0xFF8B949E),
            modifier = Modifier.padding(bottom = 10.dp)
        )

        // ── Server URL field ─────────────────────────────────────────────
        OutlinedTextField(
            value = serverUrl,
            onValueChange = onServerUrlChange,
            label = { Text("FastAPI Server URL", color = Color(0xFF8B949E)) },
            singleLine = true,
            colors = OutlinedTextFieldDefaults.colors(
                focusedBorderColor = Color(0xFF388BFD),
                unfocusedBorderColor = Color(0xFF30363D),
                focusedTextColor = Color.White,
                unfocusedTextColor = Color(0xFFE6EDF3)
            ),
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 10.dp)
        )

        // ── Camera Preview ───────────────────────────────────────────────
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
                .background(Color(0xFF161B22), RoundedCornerShape(14.dp))
                .border(1.dp, Color(0xFF30363D), RoundedCornerShape(14.dp)),
            contentAlignment = Alignment.Center
        ) {
            AndroidView(
                factory = { ctx ->
                    val previewView = PreviewView(ctx)
                    val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)

                    cameraProviderFuture.addListener({
                        val cameraProvider = cameraProviderFuture.get()

                        val preview = Preview.Builder()
                            .setTargetAspectRatio(androidx.camera.core.AspectRatio.RATIO_4_3)
                            .build().also {
                            it.setSurfaceProvider(previewView.surfaceProvider)
                        }

                        val imageAnalyzer = ImageAnalysis.Builder()
                            .setTargetAspectRatio(androidx.camera.core.AspectRatio.RATIO_4_3)
                            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                            .build()
                            .also { analysis ->
                                analysis.setAnalyzer(
                                    Executors.newSingleThreadExecutor()
                                ) { imageProxy ->
                                    onFrameReady(imageProxy)
                                }
                            }

                        // Pick best available camera:
                        // front → back → any (emulator webcam has no lens-facing metadata)
                        val cameraSelector = when {
                            cameraProvider.hasCamera(CameraSelector.DEFAULT_FRONT_CAMERA) ->
                                CameraSelector.DEFAULT_FRONT_CAMERA
                            cameraProvider.hasCamera(CameraSelector.DEFAULT_BACK_CAMERA) ->
                                CameraSelector.DEFAULT_BACK_CAMERA
                            else -> CameraSelector.Builder()
                                .addCameraFilter { cameras -> cameras.take(1) }
                                .build()
                        }

                        try {
                            cameraProvider.unbindAll()
                            cameraProvider.bindToLifecycle(
                                lifecycleOwner,
                                cameraSelector,
                                preview,
                                imageAnalyzer
                            )
                        } catch (e: Exception) {
                            e.printStackTrace()
                        }
                    }, ContextCompat.getMainExecutor(ctx))

                    previewView
                },
                modifier = Modifier.fillMaxSize()
            )
        }

        Spacer(modifier = Modifier.height(10.dp))

        // ── Status badge ─────────────────────────────────────────────────
        Box(
            modifier = Modifier
                .background(statusColor.copy(alpha = 0.15f), RoundedCornerShape(8.dp))
                .border(1.dp, statusColor.copy(alpha = 0.4f), RoundedCornerShape(8.dp))
                .padding(horizontal = 12.dp, vertical = 6.dp)
                .fillMaxWidth(),
            contentAlignment = Alignment.Center
        ) {
            Text(
                text = status,
                fontSize = 13.sp,
                fontWeight = FontWeight.Bold,
                color = statusColor
            )
        }

        Spacer(modifier = Modifier.height(8.dp))

        // ── Translation Card ─────────────────────────────────────────────
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = Color(0xFF161B22)),
            shape = RoundedCornerShape(14.dp),
            border = androidx.compose.foundation.BorderStroke(1.dp, Color(0xFF30363D))
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = word.uppercase(),
                        fontSize = 22.sp,
                        fontWeight = FontWeight.Bold,
                        color = Color.White
                    )
                    Box(
                        modifier = Modifier
                            .background(Color(0xFF21262D), CircleShape)
                            .padding(horizontal = 10.dp, vertical = 4.dp)
                    ) {
                        Text(
                            text = "${(confidence * 100).toInt()}%",
                            fontSize = 13.sp,
                            fontWeight = FontWeight.Bold,
                            color = Color(0xFF8B949E)
                        )
                    }
                }

                Spacer(modifier = Modifier.height(10.dp))

                Text(
                    text = "TRANSLATED SENTENCE",
                    fontSize = 10.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFF8B949E),
                    letterSpacing = 1.sp
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = if (sentence.isNotBlank()) sentence else "Waiting for gestures...",
                    fontSize = 20.sp,
                    fontWeight = FontWeight.SemiBold,
                    color = Color(0xFF3FB950)
                )
            }
        }
    }
}

@Composable
fun PermissionDeniedScreen(onRequestPermission: () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFF0D1117))
            .padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text("📷", fontSize = 48.sp)
        Spacer(modifier = Modifier.height(16.dp))
        Text(
            text = "Camera Permission Required",
            fontSize = 20.sp,
            fontWeight = FontWeight.Bold,
            color = Color.White
        )
        Spacer(modifier = Modifier.height(8.dp))
        Text(
            text = "ISL Speak needs camera access to recognise sign language gestures in real time.",
            fontSize = 14.sp,
            color = Color(0xFF8B949E),
            textAlign = androidx.compose.ui.text.style.TextAlign.Center
        )
        Spacer(modifier = Modifier.height(24.dp))
        Button(
            onClick = onRequestPermission,
            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF388BFD))
        ) {
            Text("Grant Camera Permission", color = Color.White)
        }
    }
}

@Composable
fun ISLSpeakTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = darkColorScheme(
            primary = Color(0xFF388BFD),
            background = Color(0xFF0D1117),
            surface = Color(0xFF161B22)
        ),
        content = content
    )
}
