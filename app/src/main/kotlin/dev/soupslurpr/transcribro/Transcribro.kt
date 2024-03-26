package dev.soupslurpr.transcribro

import androidx.annotation.StringRes
import androidx.compose.animation.animateContentSize
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideIn
import androidx.compose.animation.slideOut
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.asPaddingValues
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.VolunteerActivism
import androidx.compose.material.icons.outlined.Home
import androidx.compose.material.icons.outlined.Settings
import androidx.compose.material.icons.outlined.VolunteerActivism
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.material3.rememberTopAppBarState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Modifier
import androidx.compose.ui.input.nestedscroll.nestedScroll
import androidx.compose.ui.platform.LocalLayoutDirection
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp
import androidx.navigation.NavBackStackEntry
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.navigation
import androidx.navigation.compose.rememberNavController
import dev.soupslurpr.transcribro.ui.donate.DonateStartScreen
import dev.soupslurpr.transcribro.ui.settings.CreditsScreen
import dev.soupslurpr.transcribro.ui.settings.LicenseScreen
import dev.soupslurpr.transcribro.ui.settings.PrivacyPolicyScreen
import dev.soupslurpr.transcribro.ui.settings.SettingsStartScreen
import dev.soupslurpr.transcribro.ui.start.StartScreen
import kotlinx.coroutines.launch
import kotlin.math.absoluteValue

enum class TranscribroAppScreens(@StringRes val title: Int) {
    Start(title = R.string.start),
    StartStart(title = R.string.start),
    Settings(title = R.string.settings),
    SettingsStart(title = R.string.settings),
    SettingsLicense(title = R.string.license),
    SettingsPrivacyPolicy(title = R.string.privacy_policy),
    SettingsCredits(title = R.string.credits),
    Donate(title = R.string.donate),
    DonateStart(title = R.string.donate)
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TranscribroApp(actionApplicationPreferences: Boolean) {
    val navController = rememberNavController()

    val backStackEntry by navController.currentBackStackEntryAsState()

    val currentScreen =
        TranscribroAppScreens.valueOf(backStackEntry?.destination?.route ?: TranscribroAppScreens.Settings.name)

    val snackbarHostState = remember { SnackbarHostState() }

    val snackbarCoroutine = rememberCoroutineScope()

    val navBarSelected = if (currentScreen.name.startsWith("Start")) {
        TranscribroAppScreens.Start
    } else if (currentScreen.name.startsWith("Settings")) {
        TranscribroAppScreens.Settings
    } else if (currentScreen.name.startsWith("Donate")) {
        TranscribroAppScreens.Donate
    } else {
        currentScreen
    }

    val topAppBarScrollBehavior = TopAppBarDefaults.enterAlwaysScrollBehavior(rememberTopAppBarState())

    Scaffold(
        modifier = Modifier
            .nestedScroll(topAppBarScrollBehavior.nestedScrollConnection)
            .animateContentSize(),
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        text = stringResource(id = currentScreen.title),
                        style = MaterialTheme.typography.headlineMedium
                    )
                },
                navigationIcon = {
                    if (navController.previousBackStackEntry != null) {
                        IconButton(onClick = { navController.navigateUp() }) {
                            Icon(
                                imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                                contentDescription = "Navigate up"
                            )
                        }
                    }
                },
                windowInsets =
                WindowInsets(
                    top = ((16f + TopAppBarDefaults.windowInsets.asPaddingValues()
                        .calculateTopPadding().value) * ((topAppBarScrollBehavior.state.heightOffset / topAppBarScrollBehavior.state.heightOffsetLimit) - 1f).absoluteValue).dp,
                    bottom = ((16f + TopAppBarDefaults.windowInsets.asPaddingValues()
                        .calculateBottomPadding().value) * ((topAppBarScrollBehavior.state.heightOffset / topAppBarScrollBehavior.state.heightOffsetLimit) - 1f).absoluteValue).dp,
                    left = TopAppBarDefaults.windowInsets.asPaddingValues()
                        .calculateLeftPadding(LocalLayoutDirection.current),
                    right = TopAppBarDefaults.windowInsets.asPaddingValues()
                        .calculateRightPadding(LocalLayoutDirection.current)
                ),
                scrollBehavior = topAppBarScrollBehavior
            )
        },
        bottomBar = {
            NavigationBar {
                listOf(
                    TranscribroAppScreens.Start,
                    TranscribroAppScreens.Settings,
                    TranscribroAppScreens.Donate
                ).forEach { bottomBarScreen ->
                    NavigationBarItem(
                        selected = bottomBarScreen == navBarSelected,
                        onClick = {
                            navController.navigate(bottomBarScreen.name) {
                                popUpTo(navController.graph.findStartDestination().id) {
                                    saveState = true
                                }
                                launchSingleTop = true
                                restoreState = true
                            }
                        },
                        icon = {
                            when (bottomBarScreen) {
                                TranscribroAppScreens.Start -> Icon(
                                    imageVector = if (navBarSelected == bottomBarScreen) {
                                        Icons.Filled.Home
                                    } else {
                                        Icons.Outlined.Home
                                    },
                                    contentDescription = null
                                )

                                TranscribroAppScreens.Settings -> Icon(
                                    imageVector = if (navBarSelected == bottomBarScreen) {
                                        Icons.Filled.Settings
                                    } else {
                                        Icons.Outlined.Settings
                                    },
                                    contentDescription = null
                                )

                                TranscribroAppScreens.Donate -> Icon(
                                    imageVector = if (navBarSelected == bottomBarScreen) {
                                        Icons.Filled.VolunteerActivism
                                    } else {
                                        Icons.Outlined.VolunteerActivism
                                    },
                                    contentDescription = null
                                )

                                else -> {}
                            }
                        },
                        label = { Text(text = stringResource(id = bottomBarScreen.title)) }
                    )
                }
            }
        },
        snackbarHost = {
            SnackbarHost(snackbarHostState)
        },
    ) { innerPadding ->
        NavHost(
            navController = navController,
            startDestination = if (actionApplicationPreferences) {
                TranscribroAppScreens.Settings.name
            } else {
                TranscribroAppScreens.Start.name
            },
            modifier = Modifier.padding(innerPadding),
            enterTransition = {
                slideIn { IntOffset(it.width, 0) } + fadeIn()
            },
            exitTransition = {
                slideOut { IntOffset(-it.width, 0) } + fadeOut()
            },
            popEnterTransition = {
                slideIn { IntOffset(-it.width, 0) } + fadeIn()
            },
            popExitTransition = {
                slideOut { IntOffset(it.width, 0) } + fadeOut()
            }
        ) {
            navigation(
                route = TranscribroAppScreens.Start.name,
                startDestination = TranscribroAppScreens.StartStart.name,
                enterTransition = {
                    topAppBarScrollBehavior.state.heightOffset = 0f

                    slideIn { IntOffset(-it.width, 0) } + fadeIn()
                },
                exitTransition = {
                    slideOut { IntOffset(-it.width, 0) } + fadeOut()
                },
                popEnterTransition = {
                    if (getInitialStateNavBarRoute(initialState) == TranscribroAppScreens.Donate) {
                        slideIn { IntOffset(-it.width, 0) }
                    } else {
                        slideIn { IntOffset(-it.width, 0) }
                    } + fadeIn()
                },
                popExitTransition = {
                    slideOut { IntOffset(it.width, 0) } + fadeOut()
                }
            ) {
                composable(
                    route = TranscribroAppScreens.StartStart.name
                ) {
                    StartScreen()
                }
            }
            navigation(
                route = TranscribroAppScreens.Settings.name,
                startDestination = TranscribroAppScreens.SettingsStart.name,
                enterTransition = {
                    topAppBarScrollBehavior.state.heightOffset = 0f

                    if (getInitialStateNavBarRoute(initialState) == TranscribroAppScreens.Start) {
                        slideIn { IntOffset(it.width, 0) }
                    } else if (getInitialStateNavBarRoute(initialState) == TranscribroAppScreens.Donate) {
                        slideIn { IntOffset(-it.width, 0) }
                    } else {
                        slideIn { IntOffset(it.width, 0) }
                    } + fadeIn()
                },
                exitTransition = {
                    if (getTargetStateNavBarRoute(targetState) == TranscribroAppScreens.Start) {
                        slideOut { IntOffset(it.width, 0) }
                    } else if (getTargetStateNavBarRoute(targetState) == TranscribroAppScreens.Donate) {
                        slideOut { IntOffset(-it.width, 0) }
                    } else {
                        slideOut { IntOffset(-it.width, 0) }
                    } + fadeOut()
                },
                popEnterTransition = {
                    if (getInitialStateNavBarRoute(initialState) == TranscribroAppScreens.Start) {
                        slideIn { IntOffset(it.width, 0) }
                    } else if (getInitialStateNavBarRoute(initialState) == TranscribroAppScreens.Donate) {
                        slideIn { IntOffset(-it.width, 0) }
                    } else {
                        slideIn { IntOffset(-it.width, 0) }
                    } + fadeIn()
                },
                popExitTransition = {
                    slideOut { IntOffset(it.width, 0) } + fadeOut()
                }
            ) {
                composable(
                    route = TranscribroAppScreens.SettingsStart.name
                ) {
                    SettingsStartScreen(
                        onClickLicense = {
                            navController.navigate(TranscribroAppScreens.SettingsLicense.name)
                        },
                        onClickPrivacyPolicy = {
                            navController.navigate(TranscribroAppScreens.SettingsPrivacyPolicy.name)
                        },
                        onClickCredits = {
                            navController.navigate(TranscribroAppScreens.SettingsCredits.name)
                        }
                    )
                }
                composable(
                    route = TranscribroAppScreens.SettingsLicense.name
                ) {
                    LicenseScreen()
                }
                composable(
                    route = TranscribroAppScreens.SettingsPrivacyPolicy.name
                ) {
                    PrivacyPolicyScreen()
                }
                composable(
                    route = TranscribroAppScreens.SettingsCredits.name
                ) {
                    CreditsScreen {
                        {} // No Rust?
                    }
                }
            }
            navigation(
                route = TranscribroAppScreens.Donate.name,
                startDestination = TranscribroAppScreens.DonateStart.name,
                enterTransition = {
                    topAppBarScrollBehavior.state.heightOffset = 0f
                    if (getInitialStateNavBarRoute(initialState) == TranscribroAppScreens.Start) {
                        slideIn { IntOffset(it.width, 0) }
                    } else if (getInitialStateNavBarRoute(initialState) == TranscribroAppScreens.Settings) {
                        slideIn { IntOffset(it.width, 0) }
                    } else {
                        slideIn { IntOffset(it.width, 0) }
                    } + fadeIn()
                },
                exitTransition = {
                    if (getTargetStateNavBarRoute(targetState) == TranscribroAppScreens.Start) {
                        slideOut { IntOffset(it.width, 0) }
                    } else if (getTargetStateNavBarRoute(targetState) == TranscribroAppScreens.Settings) {
                        slideOut { IntOffset(it.width, 0) }
                    } else {
                        slideOut { IntOffset(-it.width, 0) }
                    } + fadeOut()
                },
                popEnterTransition = {
                    if (getInitialStateNavBarRoute(initialState) == TranscribroAppScreens.Start) {
                        slideIn { IntOffset(it.width, 0) }
                    } else if (getInitialStateNavBarRoute(initialState) == TranscribroAppScreens.Settings) {
                        slideIn { IntOffset(it.width, 0) }
                    } else {
                        slideIn { IntOffset(-it.width, 0) }
                    } + fadeIn()
                },
                popExitTransition = {
                    slideOut { IntOffset(it.width, 0) } + fadeOut()
                }
            ) {
                composable(
                    route = TranscribroAppScreens.DonateStart.name
                ) {
                    DonateStartScreen(
                        showSnackbarError = {
                            snackbarCoroutine.launch {
                                snackbarHostState.showSnackbar(it)
                            }
                        }
                    )
                }
            }
        }
    }
}

fun getTargetStateNavBarRoute(targetState: NavBackStackEntry): TranscribroAppScreens {
    return if (targetState.destination.route?.startsWith("Start") == true) {
        TranscribroAppScreens.Start
    } else if (targetState.destination.route?.startsWith("Settings") == true) {
        TranscribroAppScreens.Settings
    } else if (targetState.destination.route?.startsWith("Donate") == true) {
        TranscribroAppScreens.Donate
    } else {
        TranscribroAppScreens.entries.find {
            it.name == targetState.destination.route
        } ?: TranscribroAppScreens.Start
    }
}

fun getInitialStateNavBarRoute(initialState: NavBackStackEntry): TranscribroAppScreens {
    return if (initialState.destination.route?.startsWith("Start") == true) {
        TranscribroAppScreens.Start
    } else if (initialState.destination.route?.startsWith("Settings") == true) {
        TranscribroAppScreens.Settings
    } else if (initialState.destination.route?.startsWith("Donate") == true) {
        TranscribroAppScreens.Donate
    } else {
        TranscribroAppScreens.entries.find {
            it.name == initialState.destination.route
        } ?: TranscribroAppScreens.Start
    }
}