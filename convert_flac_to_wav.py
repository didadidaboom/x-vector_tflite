# convert_flac_to_wav.py
import torchaudio
signal, fs = torchaudio.load("D:/speechbrain_project/LibriSpeech/dev-clean/1272/135031/1272-135031-0000.flac")
torchaudio.save("D:/speechbrain_project/sample.wav", signal, fs)