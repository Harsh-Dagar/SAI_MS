import os
import torch
import whisper
import json

def load_model(model_name, device):
    """
    Loads the Whisper model based on the specified model name and device.
    """
    print(f"Loading {model_name} model...")
    model = whisper.load_model(model_name, device=device)
    return model

def detect_language(model, auxdio_path):
    """
    Detects the language of the audio using the Whisper model.
    """
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    language_code = max(probs, key=probs.get)
    return whisper.tokenizer.LANGUAGES[language_code].title(), language_code

def transcribe_audio(model, audio_path, options, language_code=None, task="transcribe"):
    """
    Transcribes or translates the audio to text using the Whisper model.
    """
    if task == "transcribe":
        result = whisper.transcribe(model, audio_path, **options)
    elif task == "translate":
        result = whisper.translate(model, audio_path, **options)
    
    return result["text"]

def save_results(results, output_dir, output_formats):
    """
    Saves the transcription results in the specified formats.
    """
    os.makedirs(output_dir, exist_ok=True)
    for audio_path, result in results.items():
        output_file_name = os.path.splitext(os.path.basename(audio_path))[0]

        # Saving as text file
        if "txt" in output_formats:
            with open(os.path.join(output_dir, f"{output_file_name}.txt"), 'w') as file:
                file.write(result)
            print(f"Saved result: {output_file_name}.txt")

        # Saving as JSON file
        if "json" in output_formats:
            with open(os.path.join(output_dir, f"{output_file_name}.json"), 'w') as json_file:
                json.dump({"text": result}, json_file)
            print(f"Saved result: {output_file_name}.json")

# def main():
#     """
#     The main function to run the transcription pipeline.
#     """
#     # Example usage
#     audio_path = "data/processed_audio/vad_output.wav"  # Replace with your audio file path
#     model_name = "small"  # Choose model size: tiny, base, small, medium, large-v1, large-v2
#     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using {'GPU' if DEVICE == 'cuda' else 'CPU ⚠️'}")
    
#     # Load the model
#     model = load_model(model_name, DEVICE)
    
#     # Options for transcription
#     options = {
#         'task': 'transcribe',  # or 'translate' based on your requirement
#         'verbose': True,
#         'fp16': True,
#         'best_of': 5,
#         'beam_size': 5,
#         'condition_on_previous_text': True,  # Adjust based on your requirements
#         'initial_prompt': "Transcribe the following audio",
#         'word_timestamps': False,
#     }
    
#     # Perform transcription
#     transcribed_text = transcribe_audio(model, audio_path, options)
#     print(f"Transcribed Text: {transcribed_text}")
    
#     # Save the results
#     save_results({audio_path: transcribed_text}, "output", ["txt", "json"])

# if __name__ == '__main__':
#     main()
