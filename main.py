import sys
import wave
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import webrtcvad
import torch
from src.preprocess import preprocess_audio
from src.vad import vad_detection
from src.transcription  import transcribe_audio, load_model 
from src.question_detection import identify_questions
from src.confidence_analysis import analyze_confidence
from src.speaker_isolation import isolate_speakers
from src.quality_index import calculate_quality_index

def execute_pipeline(audio_file):
    print("Pipeline execution initiated...")
    result = {}  # Initialize a dictionary to store results
    
    try:
        # Step 1: Preprocess audio
        print("\nStep 1: Preprocessing audio...")
        preprocessed_audio = "data/processed_audio/preprocessed_audio.wav"
        preprocess_audio(audio_file, preprocessed_audio)
        print(f"Step 1 Output: Preprocessed audio saved to {preprocessed_audio}.")
        print("Step 1: Audio preprocessing successful.")
        
        # Step 2: Perform VAD
        print("\nStep 2: Performing Voice Activity Detection (VAD)...")
        vad_output_audio = "data/processed_audio/vad_output.wav"
        vad_detection(preprocessed_audio, vad_output_audio)
        print(f"Step 2 Output: VAD segments saved to {vad_output_audio}.")
        print("Step 2: VAD successful.")
        
        # Step 3: Transcribe speech to text
        print("\nStep 3: Transcribing speech to text...")
        
        # Load the model and prepare options
        model = load_model('small', device="cuda" if torch.cuda.is_available() else "cpu")
        options = {
            'task': 'transcribe',
            'verbose': True,
            'fp16': True,
            'best_of': 5,
            'beam_size': 5,
            'condition_on_previous_text': True,
            'initial_prompt': None,
            'word_timestamps': False,
        }
        
        # Now call transcribe_audio with the correct parameters
        transcribed_text = transcribe_audio(model, vad_output_audio, options)
        
        print(f"Step 3 Output: Transcribed Text: \"{transcribed_text}\"")
        print("Step 3: Transcription successful.")
        
        # Step 4: Identify questions in the transcribed text
        print("\nStep 4: Identifying questions in text...")
        questions = identify_questions(transcribed_text)
        result['questions'] = questions  # Store identified questions in result
        print(f"Step 4 Output: {len(questions)} questions identified: {questions}")
        print("Step 4: Question detection successful.")
        
        # Step 5: Analyze confidence using MFCC
        print("\nStep 5: Analyzing confidence using acoustic features (MFCC)...")
        confidence = analyze_confidence(preprocessed_audio)
        result['confidence'] = confidence  # Store confidence in result
        print(f"Step 5 Output: MFCC Mean Confidence Values: {confidence}")
        print("Step 5: Confidence analysis successful.")
        
        # Step 6: Isolate speakers using MFCC and clustering
        print("\nStep 6: Isolating speakers...")
        labels, silhouette_avg = isolate_speakers(preprocessed_audio)
        result['silhouette_score'] = silhouette_avg  # Store silhouette score in result
        result['num_speakers'] = len(set(labels))  # Store the number of isolated speakers
        print(f"Step 6 Output: {len(set(labels))} speakers isolated. Silhouette Score: {silhouette_avg:.4f}")
        print("Step 6: Speaker isolation successful.")
        
        # Step 7: Calculate final quality index
        print("\nStep 7: Calculating final quality index...")
        quality_index = calculate_quality_index(questions, confidence, silhouette_avg)
        result['quality_index'] = quality_index  # Store quality index in result
        print(f"Step 7 Output: Final Quality Index: {quality_index:.4f}")
        print("Step 7: Quality index calculation successful.")
        
        print("\nPipeline execution completed successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

    return result  # Return the result dictionary


# Example execution
audio_file = "data/audio/input_file.wav"
result = execute_pipeline(audio_file)

# Output results
print("\nPipeline Results:")
print(f"Identified Questions: {result['questions']}")
print(f"Number of Questions Asked: {len(result['questions'])}")
print(f"Confidence (MFCC Mean): {result['confidence']}")
print(f"Speaker Isolation Silhouette Score: {result['silhouette_score']}")
print(f"Number of Speakers Isolated: {result['num_speakers']}")
print(f"Final Quality Index: {result['quality_index']}")
