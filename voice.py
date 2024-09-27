from vosk import Model, KaldiRecognizer
import os
import pyaudio
import difflib
from Levenshtein import ratio
import time

model = Model(r"D:\py\AI_RD\vosk-model-small-ru-0.22")  # полный путь к модели

rec = KaldiRecognizer(model, 44100)
p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=44100,
    input=True,
    frames_per_buffer=512
)
stream.start_stream()

BAN_WORDS = ["Лопата"]

while True:

    data = stream.read(30000)
    a = rec.Result() if rec.AcceptWaveform(data) else rec.PartialResult()
    a = a.strip().split('"')[3]

    for word in a.split():
        for ban in BAN_WORDS:

            if word.lower() == ban.lower():
                print("АЙ АЙ АЙ ПЛОХИЕ СЛОВЕЧКИ")
                time.sleep(10)
    a = ""