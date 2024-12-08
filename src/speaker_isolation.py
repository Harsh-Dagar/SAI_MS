from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import librosa

def isolate_speakers(audio_file):
    """
    Isolate speakers in the audio using MFCC and clustering (KMeans).
    """
    # Load audio and extract MFCC features
    y, sr = librosa.load(audio_file, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Flatten MFCC for clustering (convert time series to feature space)
    mfcc_flattened = mfcc.T  # Shape (time_steps, n_mfcc)
    
    # Cluster using KMeans (2 clusters for simplicity)
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(mfcc_flattened)
    
    # Evaluate clustering quality using silhouette score
    silhouette_avg = silhouette_score(mfcc_flattened, labels)
    
    return labels, silhouette_avg
