package dev.soupslurpr.transcribro.ui.start

import android.Manifest
import android.content.Context
import android.content.Intent
import android.provider.Settings
import android.view.inputmethod.InputMethodManager
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.requiredSize
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.res.colorResource
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.isGranted
import com.google.accompanist.permissions.rememberPermissionState
import dev.soupslurpr.transcribro.R
import dev.soupslurpr.transcribro.ui.reusablecomposables.ScreenLazyColumn
import java.time.LocalDateTime
import kotlin.random.Random

@OptIn(ExperimentalPermissionsApi::class)
@Composable
fun StartScreen() {
    val context = LocalContext.current

    val lifecycleOwner = LocalLifecycleOwner.current

    val microphonePermissionState = rememberPermissionState(
        Manifest.permission.RECORD_AUDIO
    )

    var isMyInputMethodEnabled by rememberSaveable {
        mutableStateOf(
            isMyInputMethodEnabled(context)
        )
    }

    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            if (event == Lifecycle.Event.ON_START) {
                isMyInputMethodEnabled = isMyInputMethodEnabled(context)
            }
        }

        lifecycleOwner.lifecycle.addObserver(observer)

        onDispose {
            lifecycleOwner.lifecycle.removeObserver(observer)
        }
    }

    val isAprilFools = ((LocalDateTime.now().monthValue == 4) && (LocalDateTime.now().dayOfMonth == 1))

    val randomValue = Random.nextInt(0, 24)
    val splashMessage = rememberSaveable {
        when (randomValue) {
            0 -> "Where every word counts, unless it's a typo."
            1 -> "Always here to listen, even if it's just your fridge humming."
            2 -> "Because your thoughts deserve to be transcribed, word for word."
            3 -> "Talk nerdy to me. I'll translate it to text."
            4 -> "Turning rambles into readable recaps with a tap."
            5 -> "Whisper, sing, or discuss the theory of relativity, I've got you covered."
            6 -> "Where \"I didn't catch that\" simply doesn't exist."
            7 -> "Speak of the devil, and I shall make him grammatically correct."
            8 -> "In a world of autocorrect fails, remains your faithful scribe."
            9 -> "Fine-tuning your \"ums\" and \"ahs\"—catching them when they count but skimming over when they're just filler."
            10 -> "Don't worry about the loud coffee shop. I listen to you, not the latte art."
            11 -> "Bridging the gap between brainwaves and text, one word at a time."
            12 -> "You do the talking; I handle the typing. Teamwork makes the dream work."
            13 -> "Where your voice gets VIP treatment, no velvet rope required."
            14 -> "Transcribruh, Transcribruv, Transcribrah"
            15 -> "Unofficially competing for the title of \"World’s Most Patient Listener\" since launch."
            16 -> "Because typing is soooo 20th century."
            17 -> "Quietly accurate, loudly private."
            18 -> "Speech-to-text without the cloud circus."
            19 -> "When speed matters and privacy matters more."
            20 -> "Freeing you from thumb cramps since 2024."
            21 -> "Save the thumbs, use thy tongue."
            22 -> "Doesn't work well with brainrot. Yet."
            23 -> "Making sure \"once upon a time\" never becomes \"one upon a tim\"."
            else -> "Hey! Stop reading my source code without my consent! Just kidding, I'm open source :)"
        }
    }

    ScreenLazyColumn(
        modifier = Modifier
            .fillMaxSize()
    ) {
        item {
            Column(
                modifier = Modifier.fillMaxWidth(),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Box(
                    contentAlignment = Alignment.Center,
                    modifier = Modifier
                        .size(200.dp)
                        .clip(CircleShape)
                        .background(colorResource(id = R.color.ic_launcher_background))
                ) {
                    Image(
                        painter = painterResource(R.drawable.ic_launcher_foreground),
                        contentDescription = null,
                        modifier = Modifier.requiredSize(300.dp)
                    )
                }
                Spacer(Modifier.size(16.dp))
                Text(
                    text = stringResource(
                        if (isAprilFools) {
                            R.string.welcome_april_fools
                        } else {
                            R.string.welcome
                        }
                    ),
                    style = MaterialTheme.typography.headlineLarge
                )
                Text(
                    text = splashMessage,
                    style = MaterialTheme.typography.bodySmall,
                    textAlign = TextAlign.Center
                )
            }
        }
        item {
            if (!isMyInputMethodEnabled) {
                ElevatedCard {
                    Column(
                        Modifier.padding(16.dp)
                    ) {
                        Text("If you want to use the Transcribro Voice Input keyboard, please turn it on in system settings.")
                        Spacer(Modifier.padding(8.dp))
                        FilledTonalButton(
                            onClick = {
                                val intent = Intent(Settings.ACTION_INPUT_METHOD_SETTINGS)
                                intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK
                                context.startActivity(intent)
                            }
                        ) {
                            Text("Open on-screen keyboard system settings")
                        }
                    }
                }
            }
        }
        item {
            ElevatedCard {
                Column(
                    Modifier.padding(16.dp)
                ) {
                    Text(
                        "If you want other apps that use the user-selected voice input app to use Transcribro," +
                                " you need to grant Transcribro Microphone access " +
                                "(make sure to select \"While using the app\" or it won't work properly), and then select " +
                                "Transcribro Speech Recognition Service as the voice input app in settings." +
                                " Go to System > Languages > Voice input, and then make sure Transcribro Speech Recognition Service is selected as the voice input app.",
                    )
                    Spacer(Modifier.padding(8.dp))
                    FilledTonalButton(
                        enabled = !microphonePermissionState.status.isGranted,
                        onClick = {
                            microphonePermissionState.launchPermissionRequest()
                        }
                    ) {
                        Text("Grant microphone permission (make sure to select \"While using the app\")")
                    }
                    Spacer(Modifier.padding(4.dp))
                    FilledTonalButton(
                        enabled = microphonePermissionState.status.isGranted,
                        onClick = {
                            val intent = Intent(Settings.ACTION_SETTINGS)
                            intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK
                            context.startActivity(intent)
                        }
                    ) {
                        Text("Open settings (Navigate to System > Language & region > Speech > Voice input)")
                    }
                    Spacer(Modifier.padding(8.dp))
                    Text("If you already selected Transcribro as the voice input app, please ignore this.")
                }
            }
        }
    }
}

fun isMyInputMethodEnabled(context: Context): Boolean {
    val imm = context.getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager

    val enabledInputMethods = imm.enabledInputMethodList

    val myInputMethodPackageName = context.packageName

    for (inputMethod in enabledInputMethods) {
        if (myInputMethodPackageName == inputMethod.packageName) {
            return true
        }
    }

    return false
}