# preprocess/preprocess.py
import sys
from pydub import AudioSegment
from pydub.effects import normalize
from pydub.silence import detect_nonsilent

def preprocess_audio(input_audio, output_audio):
    print(f"Preprocessing {input_audio}...")

    # Load the audio file
    audio = AudioSegment.from_file(input_audio)
    
    # Normalize volume
    normalized_audio = normalize(audio)
    
    # Detect and remove silent parts
    nonsilent_ranges = detect_nonsilent(normalized_audio, min_silence_len=500, silence_thresh=-40)
    
    if nonsilent_ranges:
        trimmed_audio = sum([normalized_audio[start:end] for start, end in nonsilent_ranges])
    else:
        print("No nonsilent segments detected. Saving normalized audio as-is.")
        trimmed_audio = normalized_audio

    # Convert to 16-bit PCM, mono, 16 kHz
    trimmed_audio = trimmed_audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    
    # Save the preprocessed audio
    trimmed_audio.export(output_audio, format="wav")
    print(f"Preprocessed audio saved to {output_audio}")

if __name__ == "__main__":
    input_audio = sys.argv[1]
    output_audio = sys.argv[2]
    preprocess_audio(input_audio, output_audio)
