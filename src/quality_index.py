import numpy as np

def calculate_quality_index(questions, confidence, silhouette_avg):
    """
    Calculate the final quality index based on multiple factors.
    """
    # Question score: number of questions identified
    question_score = len(questions)
    
    # Confidence score: mean MFCC values as a proxy for speech confidence
    confidence_score = np.mean(confidence)
    
    # Speaker quality score: silhouette score from clustering
    speaker_quality = silhouette_avg
    
    # Combine the scores (adjust weights as needed)
    quality_index = 0.4 * question_score + 0.3 * confidence_score + 0.3 * speaker_quality
    
    return quality_index
