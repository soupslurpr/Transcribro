package dev.soupslurpr.transcribro.ui.action_recognize_speech

import android.Manifest
import android.app.Activity
import android.app.Application
import android.app.PendingIntent
import android.content.ComponentName
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Mic
import androidx.compose.material3.FilledIconToggleButton
import androidx.compose.material3.Icon
import androidx.compose.material3.SnackbarDuration
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.isGranted
import com.google.accompanist.permissions.rememberPermissionState
import dev.soupslurpr.transcribro.recognitionservice.MainRecognitionService

@OptIn(ExperimentalPermissionsApi::class)
@Composable
fun ActionRecognizeSpeechScreen(
    showSnackbarError: (String, String?, Boolean, SnackbarDuration) -> Unit,
) {
    val speechRecognizerViewModel: SpeechRecognizerViewModel = viewModel()

    val speechRecognizerUiState by speechRecognizerViewModel.uiState.collectAsState()

    val context = LocalContext.current

    val microphonePermissionState = rememberPermissionState(
        Manifest.permission.RECORD_AUDIO
    )

    var alreadyRequestedMicrophonePermissionOnce by rememberSaveable {
        mutableStateOf(false)
    }

    LaunchedEffect(true) {
        if (!speechRecognizerUiState.isRecognizing) {
            if (speechRecognizerUiState.speechRecognizer.value == null) {
                speechRecognizerViewModel.setSpeechRecognizer(
                    SpeechRecognizer.createSpeechRecognizer(
                        context,
                        ComponentName(context, MainRecognitionService::class.java)
                    )
                )

                speechRecognizerViewModel.setRecognitionListener(
                    object : SpeechRecognizerViewModel.MainSpeechRecognitionListener(
                        context.applicationContext as Application,
                        speechRecognizerViewModel
                    ) {
                        override fun onResults(results: Bundle?) {
                            super.onResults(results)

                            val activity = context as Activity

                            val resultsIntent = Intent()

                            val resultsTranscripts = arrayListOf(
                                results?.getStringArrayList(
                                    SpeechRecognizer.RESULTS_RECOGNITION
                                )?.get(0) ?: ""
                            )
                            val resultsConfidenceScores = floatArrayOf(1.0F)

                            resultsIntent.putExtra(
                                RecognizerIntent.EXTRA_RESULTS,
                                resultsTranscripts,
                            )
                            resultsIntent.putExtra(RecognizerIntent.EXTRA_CONFIDENCE_SCORES, resultsConfidenceScores)

                            val resultsPendingIntentBundle =
                                activity.intent.getBundleExtra(RecognizerIntent.EXTRA_RESULTS_PENDINGINTENT_BUNDLE)
                            val pendingIntent: PendingIntent? =
                                if (Build.VERSION.SDK_INT > Build.VERSION_CODES.TIRAMISU) {
                                    activity.intent.getParcelableExtra(
                                        RecognizerIntent.EXTRA_RESULTS_PENDINGINTENT,
                                        PendingIntent::class.java
                                    )
                                } else {
                                    activity.intent.getParcelableExtra(RecognizerIntent.EXTRA_RESULTS_PENDINGINTENT)
                                }

                            when (resultsPendingIntentBundle) {
                                null -> {
                                    pendingIntent?.send(context, Activity.RESULT_OK, resultsIntent)
                                }

                                else -> {
                                    resultsPendingIntentBundle.putStringArrayList(
                                        RecognizerIntent.EXTRA_RESULTS,
                                        resultsTranscripts
                                    )
                                    resultsPendingIntentBundle.putFloatArray(
                                        RecognizerIntent.EXTRA_CONFIDENCE_SCORES,
                                        resultsConfidenceScores
                                    )

                                    pendingIntent?.send(context, Activity.RESULT_OK, resultsIntent)
                                }
                            }

                            activity.setResult(
                                Activity.RESULT_OK,
                                resultsIntent,
                            )

                            activity.finish()
                        }
                    }
                )
            }
        }
    }

    LaunchedEffect(speechRecognizerUiState.showInsufficientPermissionsError) {
        if (speechRecognizerUiState.showInsufficientPermissionsError) {
            microphonePermissionState.launchPermissionRequest()

            alreadyRequestedMicrophonePermissionOnce = true

            speechRecognizerViewModel.setShowInsufficientPermissionsError(false)
        }
    }

    LaunchedEffect(speechRecognizerUiState.showRecognizerBusyOrClientError) {
        if (speechRecognizerUiState.showRecognizerBusyOrClientError) {
            showSnackbarError(
                "Recognition is finishing, please wait or cancel.",
                null,
                true,
                SnackbarDuration.Short
            )

            speechRecognizerViewModel.setShowRecognizerBusyOrClientError(false)
        }
    }

    LaunchedEffect(microphonePermissionState.status.isGranted) {
        if (!speechRecognizerUiState.isRecognizing && (!alreadyRequestedMicrophonePermissionOnce || microphonePermissionState.status.isGranted)) {
            speechRecognizerViewModel.startListening(
                Intent()
            )
        }
    }

    FilledIconToggleButton(
        checked = speechRecognizerUiState.isRecognizing,
        onCheckedChange = {
            if (speechRecognizerUiState.isRecognizing) {
                speechRecognizerViewModel.stopListening()
            } else {
                speechRecognizerViewModel.startListening(
                    Intent()
                )
            }
        },
        modifier = Modifier
            .fillMaxHeight()
            .fillMaxWidth(),
        shape = RoundedCornerShape(10.dp)
    ) {
        Icon(
            imageVector = Icons.Outlined.Mic,
            contentDescription = if (speechRecognizerUiState.isRecognizing) {
                "Speech recognition active"
            } else {
                "Speech recognition inactive"
            },
            modifier = Modifier.size(165.dp)
        )
    }
}
