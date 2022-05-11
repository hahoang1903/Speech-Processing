from os import path, walk, makedirs
from pathlib import Path
import librosa
from scipy.io import savemat, loadmat
import numpy as np


def extract_mfccs(audio_dir, mfcc_dir):
    makedirs(mfcc_dir, exist_ok=True)

    for _, dirs, _ in walk(audio_dir):
        for dir in dirs:
            word_dir = path.join(mfcc_dir, dir)
            makedirs(word_dir, exist_ok=True)

            for _, _, files in walk(path.join(audio_dir, dir)):
                for file in files:
                    try:
                        signal, sr = librosa.load(
                            path.join(audio_dir, dir, file))
                        S = librosa.feature.melspectrogram(
                            y=signal, sr=sr, n_fft=512, n_mels=40, hop_length=256)
                        mfccs = librosa.feature.mfcc(S=S, n_mfcc=13)
                        features = np.concatenate(
                            [mfccs, librosa.feature.delta(
                                mfccs), librosa.feature.delta(mfccs, order=2)]
                        )

                        savemat(path.join(word_dir, Path(file).stem),
                                {'features': features})
                    except:
                        continue


def get_mfccs(mfcc_dir):
    mfccs = []

    for _, dirs, _ in walk(mfcc_dir):
        for dir in dirs:
            for subdir, _, files in walk(path.join(mfcc_dir, dir)):
                mfccs.append(
                    {
                        'label': dir,
                        'mfccs': [
                            loadmat(path.join(subdir, file))['features'] for file in files
                        ]
                    }
                )
    return mfccs
