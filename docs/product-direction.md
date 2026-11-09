# Product direction

Transcribro provides private speech input where people already write. Its
primary interface is a voice keyboard with correction controls and an easy
return to the user's typing keyboard. Apps can also invoke its Android speech
recognition interfaces. Audio-file transcription is a secondary conversion
tool; BeauTyXT owns the dedicated writing workspace.

## Platform and privacy

- Support Android 17 or newer on 64-bit ARM and x86 devices.
- Run recognition entirely on the device, with no Internet permission,
  account, analytics, or model downloads in the installed app.
- Bundle the maintained English recognizer. Preserve original Medium model
  accuracy when optimizing its runtime or storage.
- Persist settings only. Audio, transcripts, and editor context remain in
  memory while needed; there is no recording or transcript library.
- Share or save results only to a destination the user chooses. Unfinished
  work has no automatic recovery after process death.
- Explain that receiving apps can retain inserted text, including tentative
  live text, under their own policies.

## Dictation

Opening the voice keyboard starts listening by default after microphone
permission has been granted. Users can disable automatic start. Password and
PIN fields require an explicit Start action and receive completed text without
tentative previews. Opening Home or Settings never starts listening.

Finish and Cancel leave the visible keyboard idle until another Start action
or keyboard opening. Hiding the keyboard or leaving its destination field
stops listening. Ordinary thinking pauses do not end a default session.
Inactivity and session bounds stop capture with feedback, preserve accepted
speech, and do not initiate sending. Overload also finishes accepted audio;
dropping queued speech is not an acceptable recovery strategy.

Live text revises the unfinished phrase while keeping completed phrases
stable. Editors that cannot reliably support composition receive completed
phrases. Disabling Live text also skips tentative recognition work. Dictation
preserves wording with punctuation and capitalization; rewriting, summarizing,
and spoken editing commands are outside the product's scope. Editor context
supports insertion and correction, but does not enter recognition.

## Editing and sending

Keep undo, redo, repeating backspace, separate Return and editor-action keys,
selection replacement, Keep selected text/Clear field, settings, and keyboard
switching accessible. Replacement respects surrounding spacing, case, and
punctuation. Correction during dictation completes the captured phrase before
editing, then continues in the same field. Unexpected field changes stop
dictation so words cannot enter another destination.

Manual Send completes the speech captured before the requested boundary,
sends completed text, and continues listening in the same field. Tentative
text must never trigger Send.

Automatic start, Finish at a pause, and return to the previous keyboard are
independent settings. Automatic start defaults on; the other two default off.
The configurable pause is one, two, or three seconds, defaulting to three.

Auto-send defaults off and offers two timings:

- After dictation finishes: send once after successful completion.
- After each phrase: finalize and send at a speech pause, then keep listening.

The enabled timing is visible in the keyboard header and opens its setting
when tapped. Enabling Auto-send removes the need for confirmation on each
message. A same-package v7 upgrade preserves an enabled Auto-send preference
as After each phrase; an existing Reimagined preference takes precedence.

## Audio-file transcription

Support voice messages and personal recordings with a temporary result view,
selectable text, source playback and seeking, and Copy, Share, and Save.
Editing happens in the receiving app. Speaker labels, synchronized word
highlighting, meeting tooling, and a transcript archive are outside the scope.

Transcription continues when the app is backgrounded or the screen is locked,
with a progress notification offering Stop. Live dictation takes priority and
can pause file processing. Enforce the 30-minute file bound on both metadata
and decoded samples. Format availability follows Android's installed decoders.

## Interface and accessibility

Home provides voice-input setup, readiness checks, a temporary dictation test
field, settings, and file transcription. Explain Android's separate voice
input service and keyboard selections. Keep privacy information, source,
license notices, and support available without an acceptance step or donation
prompts.

Use Material 3 Expressive, dynamic color, and the system's light/dark setting,
in the same visual family as BeauTyXT. Keep the keyboard compact while
preserving accessible targets and access to all correction controls. Landscape
can place voice and editing controls side by side.

Starting, correcting, cancelling, and finishing need accessible paths through
touch, screen readers, and switch access. Listening sounds and haptics are
optional, follow actual capture readiness and shutdown, and respect Android's
sound and haptic settings. Hold-to-repeat gives one haptic cue when repetition
begins. Automatic sending supports users who benefit from fewer interactions.

Implementation ownership and resource bounds are in [architecture.md](architecture.md).
