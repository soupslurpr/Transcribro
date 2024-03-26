package dev.soupslurpr.transcribro.ui.settings

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import dev.soupslurpr.transcribro.R

@Composable
fun PrivacyPolicyScreen() {
    Column(
        modifier = Modifier.verticalScroll(rememberScrollState())
    ) {
        SelectionContainer {
            Text(
                modifier = Modifier.padding(horizontal = 15.dp),
                text = stringResource(R.string.full_privacy_policy)
            )
        }
    }
}