package dev.soupslurpr.transcribro

import androidx.annotation.StringRes
import androidx.compose.animation.AnimatedContentScope
import androidx.compose.animation.AnimatedContentTransitionScope
import androidx.compose.animation.EnterTransition
import androidx.compose.animation.ExitTransition
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
import androidx.navigation.NamedNavArgument
import androidx.navigation.NavBackStackEntry
import androidx.navigation.NavDeepLink
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.NavGraphBuilder
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

enum class TranscribroAppScreens(@StringRes val title: Int) {
    Start(title = R.string.start),
    StartStart(title = R.string.start),
    Settings(title = R.string.settings),
    SettingsStart(title = R.string.settings),
    SettingsLicense(title = R.string.license),
    SettingsPrivacyPolicy(title = R.string.privacy_policy),
    SettingsCredits(title = R.string.credits),
    Donate(title = R.string.donate),
    DonateStart(title = R.string.donate),
}

val navBarScreens = listOf(
    TranscribroAppScreens.Start,
    TranscribroAppScreens.Settings,
    TranscribroAppScreens.Donate,
)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TranscribroApp(actionApplicationPreferences: Boolean) {
    val navController = rememberNavController()

    val backStackEntry by navController.currentBackStackEntryAsState()

    val currentScreen =
        TranscribroAppScreens.valueOf(backStackEntry?.destination?.route ?: TranscribroAppScreens.Settings.name)

    val snackbarHostState = remember { SnackbarHostState() }

    val snackbarCoroutine = rememberCoroutineScope()

    val navBarSelected = navBarScreens.find {
        currentScreen.name.startsWith(it.name)
    }

    val topAppBarScrollBehavior = TopAppBarDefaults.pinnedScrollBehavior(rememberTopAppBarState())

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
                scrollBehavior = topAppBarScrollBehavior
            )
        },
        bottomBar = {
            if (navBarSelected != null) {
                NavigationBar {
                    navBarScreens.forEach { navBarScreen ->
                        NavigationBarItem(
                            selected = navBarScreen == navBarSelected,
                            onClick = {
                                navController.navigate(navBarScreen.name) {
                                    popUpTo(navController.graph.findStartDestination().id) {
                                        saveState = true
                                    }
                                    launchSingleTop = true
                                    restoreState = true
                                }
                            },
                            icon = {
                                when (navBarScreen) {
                                    TranscribroAppScreens.Start -> Icon(
                                        imageVector = if (navBarSelected == navBarScreen) {
                                            Icons.Filled.Home
                                        } else {
                                            Icons.Outlined.Home
                                        },
                                        contentDescription = null
                                    )

                                    TranscribroAppScreens.Settings -> Icon(
                                        imageVector = if (navBarSelected == navBarScreen) {
                                            Icons.Filled.Settings
                                        } else {
                                            Icons.Outlined.Settings
                                        },
                                        contentDescription = null
                                    )

                                    TranscribroAppScreens.Donate -> Icon(
                                        imageVector = if (navBarSelected == navBarScreen) {
                                            Icons.Filled.VolunteerActivism
                                        } else {
                                            Icons.Outlined.VolunteerActivism
                                        },
                                        contentDescription = null
                                    )

                                    else -> {}
                                }
                            },
                            label = { Text(text = stringResource(id = navBarScreen.title)) }
                        )
                    }
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
            navigationWithDefaultSlideTransitions(
                route = TranscribroAppScreens.Start,
                startDestination = TranscribroAppScreens.StartStart.name,
            ) {
                composableWithDefaultSlideTransitions(
                    route = TranscribroAppScreens.StartStart
                ) {
                    StartScreen()
                }
            }
            navigationWithDefaultSlideTransitions(
                route = TranscribroAppScreens.Settings,
                startDestination = TranscribroAppScreens.SettingsStart.name,
            ) {
                composableWithDefaultSlideTransitions(
                    route = TranscribroAppScreens.SettingsStart
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
                composableWithDefaultSlideTransitions(
                    route = TranscribroAppScreens.SettingsLicense
                ) {
                    LicenseScreen()
                }
                composableWithDefaultSlideTransitions(
                    route = TranscribroAppScreens.SettingsPrivacyPolicy
                ) {
                    PrivacyPolicyScreen()
                }
                composableWithDefaultSlideTransitions(
                    route = TranscribroAppScreens.SettingsCredits
                ) {
                    CreditsScreen {
                        {} // No Rust?
                    }
                }
            }
            navigationWithDefaultSlideTransitions(
                route = TranscribroAppScreens.Donate,
                startDestination = TranscribroAppScreens.DonateStart.name,
            ) {
                composableWithDefaultSlideTransitions(
                    route = TranscribroAppScreens.DonateStart
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

fun getStateNavBarRoute(state: NavBackStackEntry): TranscribroAppScreens? {
    state.destination.route?.let { return TranscribroAppScreens.valueOf(it) }
    return null
}

fun getEnterTransition(initialState: NavBackStackEntry, targetState: NavBackStackEntry): EnterTransition {
    val initialNavBarRoute = getStateNavBarRoute(initialState)
    val targetNavBarRoute = getStateNavBarRoute(targetState)

    return if ((initialNavBarRoute != null) && (targetNavBarRoute != null)) {
        slideIn {
            IntOffset(
                if (initialNavBarRoute.ordinal > targetNavBarRoute.ordinal) {
                    -it.width
                } else {
                    it.width
                },
                0
            )
        } + fadeIn()
    } else {
        EnterTransition.None
    }
}

fun getExitTransition(initialState: NavBackStackEntry, targetState: NavBackStackEntry): ExitTransition {
    val initialNavBarRoute = getStateNavBarRoute(initialState)
    val targetNavBarRoute = getStateNavBarRoute(targetState)

    return if ((initialNavBarRoute != null) && (targetNavBarRoute != null)) {
        slideOut {
            IntOffset(
                if (initialNavBarRoute.ordinal > targetNavBarRoute.ordinal) {
                    it.width
                } else {
                    -it.width
                },
                0
            )
        } + fadeOut()
    } else {
        ExitTransition.None
    }
}

fun NavGraphBuilder.composableWithDefaultSlideTransitions(
    route: TranscribroAppScreens,
    arguments: List<NamedNavArgument> = emptyList(),
    deepLinks: List<NavDeepLink> = emptyList(),
    enterTransition: @JvmSuppressWildcards (AnimatedContentTransitionScope<NavBackStackEntry>.() -> EnterTransition?)? = null,
    exitTransition: @JvmSuppressWildcards (AnimatedContentTransitionScope<NavBackStackEntry>.() -> ExitTransition?)? = null,
    popEnterTransition: @JvmSuppressWildcards (AnimatedContentTransitionScope<NavBackStackEntry>.() -> EnterTransition?)? = enterTransition,
    popExitTransition: @JvmSuppressWildcards (AnimatedContentTransitionScope<NavBackStackEntry>.() -> ExitTransition?)? = exitTransition,
    content: @Composable (AnimatedContentScope.(NavBackStackEntry) -> Unit)
) {
    composable(
        route.name,
        arguments,
        deepLinks,
        if (enterTransition == null) {
            {
                getEnterTransition(initialState, targetState)
            }
        } else {
            null
        },
        if (exitTransition == null) {
            {
                getExitTransition(initialState, targetState)
            }
        } else {
            null
        },
        if (popEnterTransition == null) {
            {
                getEnterTransition(initialState, targetState)
            }
        } else {
            null
        },
        if (popExitTransition == null) {
            {
                getExitTransition(initialState, targetState)
            }
        } else {
            null
        },
        content
    )
}

fun NavGraphBuilder.navigationWithDefaultSlideTransitions(
    startDestination: String,
    route: TranscribroAppScreens,
    arguments: List<NamedNavArgument> = emptyList(),
    deepLinks: List<NavDeepLink> = emptyList(),
    enterTransition: @JvmSuppressWildcards (AnimatedContentTransitionScope<NavBackStackEntry>.() -> EnterTransition?)? =
        null,
    exitTransition: (AnimatedContentTransitionScope<NavBackStackEntry>.() -> ExitTransition?)? =
        null,
    popEnterTransition: (
    AnimatedContentTransitionScope<NavBackStackEntry>.() -> EnterTransition?
    )? = enterTransition,
    popExitTransition: (
    AnimatedContentTransitionScope<NavBackStackEntry>.() -> ExitTransition?
    )? = exitTransition,
    builder: NavGraphBuilder.() -> Unit
) {
    navigation(
        startDestination,
        route.name,
        arguments,
        deepLinks,
        if (enterTransition == null) {
            {
                getEnterTransition(initialState, targetState)
            }
        } else {
            null
        },
        if (exitTransition == null) {
            {
                getExitTransition(initialState, targetState)
            }
        } else {
            null
        },
        if (popEnterTransition == null) {
            {
                getEnterTransition(initialState, targetState)
            }
        } else {
            null
        },
        if (popExitTransition == null) {
            {
                getExitTransition(initialState, targetState)
            }
        } else {
            null
        },
        builder
    )
}