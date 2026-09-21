package com.isl.speak.network

import com.google.gson.annotations.SerializedName
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Response
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.POST
import java.util.concurrent.TimeUnit

data class PredictRequest(
    @SerializedName("sequence") val sequence: List<List<Float>>,
    @SerializedName("model_type") val modelType: String = "lstm"
)

data class PredictResponse(
    @SerializedName("label") val label: String,
    @SerializedName("confidence") val confidence: Float,
    @SerializedName("class_index") val classIndex: Int,
    @SerializedName("status") val status: String,
    @SerializedName("raw_label") val rawLabel: String
)

data class HealthResponse(
    @SerializedName("status") val status: String,
    @SerializedName("device") val device: String,
    @SerializedName("num_classes") val numClasses: Int
)

interface PredictApiService {
    @GET("health")
    suspend fun checkHealth(): Response<HealthResponse>

    @POST("predict")
    suspend fun predictGesture(@Body request: PredictRequest): Response<PredictResponse>
}

object NetworkClient {
    private var retrofit: Retrofit? = null
    private var currentBaseUrl: String = "http://10.0.2.2:8000/"

    fun getService(baseUrl: String = currentBaseUrl): PredictApiService {
        val formattedUrl = if (baseUrl.endsWith("/")) baseUrl else "$baseUrl/"
        if (retrofit == null || currentBaseUrl != formattedUrl) {
            currentBaseUrl = formattedUrl
            val interceptor = HttpLoggingInterceptor().apply {
                level = HttpLoggingInterceptor.Level.BASIC
            }
            val okHttpClient = OkHttpClient.Builder()
                .addInterceptor(interceptor)
                .connectTimeout(5, TimeUnit.SECONDS)
                .readTimeout(10, TimeUnit.SECONDS)
                .writeTimeout(10, TimeUnit.SECONDS)
                .build()

            retrofit = Retrofit.Builder()
                .baseUrl(formattedUrl)
                .client(okHttpClient)
                .addConverterFactory(GsonConverterFactory.create())
                .build()
        }
        return retrofit!!.create(PredictApiService::class.java)
    }
}
