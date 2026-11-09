package dev.soupslurpr.transcribro;

import android.content.ContentProvider;
import android.content.ContentValues;
import android.database.Cursor;
import android.database.MatrixCursor;
import android.net.Uri;
import android.os.ParcelFileDescriptor;
import android.provider.OpenableColumns;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

/** Standalone test-APK provider, without dependencies supplied by the instrumented app. */
public final class AudioFixtureProvider extends ContentProvider {
    @Override
    public boolean onCreate() { return true; }

    private String name(Uri uri) {
        String name = uri.getLastPathSegment();
        if (!"dictation-stereo.wav".equals(name) && !"dictation-stereo.m4a".equals(name)
                && !"invalid-audio.txt".equals(name)) {
            throw new IllegalArgumentException("Unknown fixture");
        }
        return name;
    }

    @Override
    public String getType(Uri uri) {
        String name = name(uri);
        return name.endsWith("wav") ? "audio/wav" : name.endsWith("m4a") ? "audio/mp4" : "text/plain";
    }

    @Override
    public ParcelFileDescriptor openFile(Uri uri, String mode) throws FileNotFoundException {
        if (!"r".equals(mode)) throw new FileNotFoundException("Read only");
        File file = null;
        try {
            file = File.createTempFile("audio-fixture-", null, getContext().getCacheDir());
            try (InputStream source = getContext().getAssets().open(name(uri));
                    FileOutputStream output = new FileOutputStream(file)) {
                source.transferTo(output);
            }
            return ParcelFileDescriptor.open(file, ParcelFileDescriptor.MODE_READ_ONLY);
        } catch (IOException failure) {
            throw new FileNotFoundException("Could not open fixture");
        } finally {
            // The returned descriptor remains valid after unlinking the synthetic fixture.
            if (file != null) file.delete();
        }
    }

    @Override
    public Cursor query(Uri uri, String[] projection, String selection, String[] args, String order) {
        String[] columns = projection == null
                ? new String[] {OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE} : projection;
        MatrixCursor result = new MatrixCursor(columns);
        Object[] values = new Object[columns.length];
        for (int index = 0; index < columns.length; index++) {
            if (OpenableColumns.DISPLAY_NAME.equals(columns[index])) values[index] = name(uri);
        }
        result.addRow(values);
        return result;
    }

    @Override
    public Uri insert(Uri uri, ContentValues values) { throw new UnsupportedOperationException(); }
    @Override
    public int update(Uri uri, ContentValues values, String selection, String[] args) { throw new UnsupportedOperationException(); }
    @Override
    public int delete(Uri uri, String selection, String[] args) { throw new UnsupportedOperationException(); }
}
