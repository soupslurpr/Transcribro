package dev.soupslurpr.transcribro.ui.donate

import android.content.ActivityNotFoundException
import android.content.Intent
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Share
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.ExperimentalTextApi
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.UrlAnnotation
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.app.ActivityOptionsCompat
import androidx.core.content.ContextCompat
import dev.soupslurpr.transcribro.R
import dev.soupslurpr.transcribro.ui.reusablecomposables.ClickableText
import dev.soupslurpr.transcribro.ui.reusablecomposables.ScreenLazyColumn

@OptIn(ExperimentalTextApi::class)
@Composable
fun DonateStartScreen(
    showSnackbarError: (String) -> Unit
) {
    val localUriHandler = LocalUriHandler.current

    var showAlertDialog by rememberSaveable { mutableStateOf(false) }

    val context = LocalContext.current

    val title = "Monero"

    val qrCodePainterResourceId = R.drawable.donation_monero_address_qr_code

    val qrCodeContentDescription = "Monero donation qr code"

    val address = "88rAaNowhaC8JG8NJDpcdRWr1gGVmtFPnHWPS9xXvqY44G4XKVi5hZMax2FQ6B8KAcMpzkeJAhNek8qMHZjjwvkEKuiyBKF"

    val addressUrl =
        "monero:$address" +
                "?recipient_name=soupslurpr&tx_description=Donation%20to%20soupslurpr"

    ScreenLazyColumn(
        modifier = Modifier
            .fillMaxSize()
    ) {
        item {
            Text(
                "Enjoy Transcribro? You can donate to soupslurpr, the lead developer of Transcribro to support their " +
                        "work on Transcribro and their other open source projects. Thank you!"
            )
        }
        item {
            ElevatedCard(Modifier.fillMaxWidth()) {
                Column(
                    Modifier
                        .fillMaxWidth()
                        .padding(16.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        title,
                        Modifier
                            .padding(bottom = 24.dp)
                            .align(Alignment.Start),
                        style = MaterialTheme.typography.titleLarge
                    )
                    Box(
                        modifier = Modifier
                            .padding(bottom = 16.dp)
                            .clip(RoundedCornerShape(20.dp))
                            .background(MaterialTheme.colorScheme.primaryContainer)
                    ) {
                        Icon(
                            painter = painterResource(id = qrCodePainterResourceId),
                            contentDescription = qrCodeContentDescription,
                            modifier = Modifier
                                .size(200.dp)
                                .align(Alignment.Center)
                                .clickable(
                                    onClickLabel = "enlarge image",
                                    role = Role.Image
                                ) {
                                    showAlertDialog = true
                                },
                            tint = MaterialTheme.colorScheme.onPrimaryContainer
                        )
                    }

                    val annotatedString = buildAnnotatedString {
                        pushUrlAnnotation(UrlAnnotation(addressUrl))
                        pushStringAnnotation("URL", addressUrl)
                        pushStyle(SpanStyle(color = MaterialTheme.colorScheme.primary, fontWeight = FontWeight.Bold))

                        append(address)

                        pop()
                        pop()
                        pop()
                    }

                    ClickableText(
                        text = annotatedString,
                        onClick = { offset ->
                            annotatedString
                                .getStringAnnotations("URL", offset, offset).firstOrNull()
                                ?.let { annotation ->
                                    try {
                                        localUriHandler.openUri(annotation.item)
                                    } catch (e: IllegalStateException) {
                                        showSnackbarError("Couldn't find an app to open donation address with!")
                                    } catch (e: ActivityNotFoundException) {
                                        showSnackbarError("Couldn't find an app to open donation address with!")
                                    }
                                }
                        },
                    )
                    IconButton(
                        onClick = {
                            var sendIntent = Intent()

                            sendIntent = sendIntent.apply {
                                action = Intent.ACTION_SEND
                                putExtra(Intent.EXTRA_TEXT, address)
                                type = "text/plain"
                            }

                            val shareIntent = Intent.createChooser(sendIntent, null)
                            ContextCompat.startActivity(
                                context,
                                shareIntent,
                                ActivityOptionsCompat.makeBasic().toBundle()
                            )
                        }
                    ) {
                        Icon(Icons.Filled.Share, contentDescription = "Share")
                    }
                }
                if (showAlertDialog) {
                    AlertDialog(
                        onDismissRequest = { showAlertDialog = false },
                        confirmButton = {
                            TextButton(
                                onClick = { showAlertDialog = false }
                            ) {
                                Text("OK")
                            }
                        },
                        text = {
                            Box(
                                modifier = Modifier
                                    .clip(RoundedCornerShape(20.dp))
                                    .background(MaterialTheme.colorScheme.primaryContainer)
                            ) {
                                Icon(
                                    painter = painterResource(id = qrCodePainterResourceId),
                                    contentDescription = "$qrCodeContentDescription (enlarged)",
                                    modifier = Modifier
                                        .aspectRatio(1.0f)
                                        .align(Alignment.Center),
                                    tint = MaterialTheme.colorScheme.onPrimaryContainer
                                )
                            }
                        }
                    )
                }
            }
        }
    }
}