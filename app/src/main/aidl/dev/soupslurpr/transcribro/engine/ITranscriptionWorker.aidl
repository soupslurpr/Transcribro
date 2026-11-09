package dev.soupslurpr.transcribro.engine;

import android.os.ParcelFileDescriptor;
import dev.soupslurpr.transcribro.engine.ITranscriptionEvents;

interface ITranscriptionWorker {
    oneway void start(in ParcelFileDescriptor audio, boolean stopOnSilence, boolean realtime, boolean bufferedCapture, boolean previews, int pauseMillis, ITranscriptionEvents events);
    oneway void startFile(in ParcelFileDescriptor pcm, ITranscriptionEvents events);
    oneway void setPaused(boolean paused);
    void finishPhrase(long request, long afterSamples);
    void captureOverloaded();
    oneway void cancel();
}
