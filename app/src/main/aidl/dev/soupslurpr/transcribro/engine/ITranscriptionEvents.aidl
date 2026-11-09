package dev.soupslurpr.transcribro.engine;

oneway interface ITranscriptionEvents {
    void ready();
    void speech(boolean speaking);
    void level(float rms);
    void progress(long completedMs, long durationMs);
    void inputEnded(int reason);
    void segment(String text, boolean phraseEnd);
    void hypothesis(String text);
    void phraseFinished(long request);
    void finished();
    void failed(int reason);
}
