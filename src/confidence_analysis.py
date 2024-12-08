import librosa
import numpy as np

def analyze_confidence(audio_file):
    """
    Analyze speech confidence using MFCC (Mel-Frequency Cepstral Coefficients).
    """
    # Load audio file (mono, 16kHz)
    y, sr = librosa.load(audio_file, sr=16000)
    
    # Extract MFCCs from the audio signal
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Compute the mean of MFCCs for each coefficient (summarize features)
    mfcc_mean = np.mean(mfcc, axis=1)
    
    return mfcc_mean
