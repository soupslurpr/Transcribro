"""Small benchmark adapter; all streaming and inference run in native code."""

import ctypes as ct

import numpy as np


class Pipeline:
    def __init__(self, library, archive, tokenizer):
        self.library = ct.CDLL(str(library))
        self.library.ts_create.argtypes = [
            ct.c_char_p,
            ct.c_char_p,
            ct.c_void_p,
            ct.c_size_t,
        ]
        self.library.ts_create.restype = ct.c_void_p
        self.library.ts_destroy.argtypes = [ct.c_void_p]
        self.library.ts_push.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t]
        self.library.ts_update.argtypes = [ct.c_void_p, ct.c_int]
        self.library.ts_reset.argtypes = [ct.c_void_p, ct.c_int]
        self.library.ts_text.argtypes = [ct.c_void_p]
        self.library.ts_text.restype = ct.c_char_p
        self.library.ts_error.restype = ct.c_char_p
        error = ct.create_string_buffer(1024)
        self.handle = self.library.ts_create(
            str(archive).encode(), str(tokenizer).encode(), error, len(error)
        )
        if not self.handle:
            raise RuntimeError(error.value.decode())

    def checked(self, status):
        if status:
            raise RuntimeError(self.library.ts_error().decode())

    def close(self):
        if self.handle:
            self.library.ts_destroy(self.handle)
            self.handle = None

    def transcribe(self, audio, previews=True):
        if not previews:
            raise ValueError("native qualification requires app preview policy")
        self.checked(self.library.ts_reset(self.handle, 1))
        audio = np.pad(np.asarray(audio, "f4"), (0, -len(audio) % 512 + 8192))
        ends = list(range(8192, audio.size - 8192 + 1, 8192)) + [audio.size]
        pushed = analyzed = 0
        for end in ends:
            chunk = np.ascontiguousarray(audio[pushed:end])
            self.checked(
                self.library.ts_push(self.handle, chunk.ctypes.data, chunk.size)
            )
            pushed = end
            final = end == audio.size
            if end - analyzed >= 16000 or final:
                self.checked(self.library.ts_update(self.handle, int(final)))
                analyzed = end
        return self.library.ts_text(self.handle).decode()
