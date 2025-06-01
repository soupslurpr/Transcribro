# Contributing

Thanks for your interest in contributing!

If you want to suggest a feature or notify us about a bug, please use the issue tracker.

This project uses Git submodules. Clone it using the `--recurse-submodules` option:

```bash
git clone --recurse-submodules https://github.com/soupslurpr/Transcribro
```

Before working on a feature, please make sure to discuss the planned implementation in the issue for the feature and get approval from @soupslurpr to ensure it meets the project's requirements.

If you need help with development or have questions, it's recommended to join the Transcribro space on matrix at
https://matrix.to/#/#transcribro:matrix.org and join the Transcribro General room and ask for help there from
[soupslurpr](https://github.com/soupslurpr), the lead developer.

As of now, translations are not accepted.

Here are some things to know so that your time isn't potentially wasted.
Transcribro currently depends on whisper.cpp to run the OpenAI Whisper models. You need C++ tooling installed
to compile. There are plans to move to using Rust to run the models once a machine learning library that can run
Whisper as fast as whisper.cpp is available.

[//]: # (Transcribro has some Rust code that gets compiled into a library.)

[//]: # (The source code can be found at the transcribro_rs directory.)

[//]: # (Look at useful-commands.txt for useful commands and info that will probably help with building it.)

[//]: # ()

[//]: # (Java code is not accepted, we will only use Kotlin and Rust if needed. Unsafe Rust code should be avoided, but if)

[//]: # (there is truly no other way, it will be heavily scrutinized.)

Views should be avoided, and only Jetpack Compose should be used unless there is no other way, but it
has to be vital.
