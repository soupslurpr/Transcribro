# Transcribro

Transcribro is a private and on-device speech recognition keyboard and service for Android.\
It uses whisper.cpp to run the OpenAI Whisper family of models and Silero VAD for voice activity detection.\
It features a voice input keyboard, enabling you to type with speech.\
It can also be used by other apps either explicitly or when set as the user-selected speech to text app which some apps
may use for speech to text.

## Language support

Transcribro currently only supports English. However, supporting other languages is planned and tracked in https://github.com/soupslurpr/Transcribro/issues/18.

## Download

Transcribro is available on the [Accrescent](https://accrescent.app) app store and GitHub releases.\
[Accrescent](https://accrescent.app) is the recommended way to get Transcribro as it is more secure
than GitHub releases.\
Click on the badge below to get it on [Accrescent](https://accrescent.app).

<a href="https://accrescent.app/app/dev.soupslurpr.transcribro">
    <img alt="Get it on Accrescent" src="https://accrescent.app/badges/get-it-on.png" height="80">
</a>

The package name and SHA-256 hash of the signing certificate is below, so if you are downloading the APK, you can
verify Transcribro with [`apksigner`](https://developer.android.com/studio/command-line/apksigner#usage-verify)
using `apksigner verify --print-certs Transcribro-X.Y.Z.apk` and/or
[AppVerifier](https://github.com/soupslurpr/AppVerifier).
If you are downloading from [Accrescent](https://accrescent.app) then you should verify
[Accrescent](https://accrescent.app) itself [here](https://accrescent.app/faq#verifying).

dev.soupslurpr.transcribro\
7D:BC:FB:FA:A1:35:B4:4E:6E:93:91:02:25:DC:B1:4E:05:82:91:DA:8C:2D:36:22:73:49:49:B7:1A:B3:BE:64

It can also be found on a [Bluesky post](https://bsky.app/profile/soupslurpr.dev/post/3kopox4ffl72t)
to distrust the website.
It is encouraged to verify it's the same with other people as well for assurance.

## Community

Join the Matrix space at https://matrix.to/#/#transcribro:matrix.org for the General, Announcements, and
Testing rooms.

## Contributing

Check [CONTRIBUTING.md](https://github.com/soupslurpr/Transcribro/blob/master/CONTRIBUTING.md) for things to know
if you want to contribute.

## Donation

Thank you to everyone who donated.

## Screenshots

<img src="/Screenshot_20250214-223133.png" alt="Screenshot of the keyboard UI, focused on the search bar of Vanadium's incognito tab." width="250">
