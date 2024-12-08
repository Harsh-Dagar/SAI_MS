# vad/vad.py
import sys
import wave
import webrtcvad
from pydub import AudioSegment

def vad_detection(input_audio, output_audio):
    print(f"Performing VAD on {input_audio}...")

    # Ensure audio is 16-bit PCM, mono, 16 kHz
    audio = AudioSegment.from_file(input_audio)
    audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    temp_audio_path = "temp_audio.wav"
    audio.export(temp_audio_path, format="wav")

    vad = webrtcvad.Vad()
    vad.set_mode(3)  # Set aggressiveness level (1-3, higher is more aggressive)

    with wave.open(temp_audio_path, 'rb') as wf:
        sample_rate = wf.getframerate()
        audio_frames = wf.readframes(wf.getnframes())
        frame_duration_ms = 30  # Duration of each frame in ms (default: 30 ms)
        frame_size = int(sample_rate * frame_duration_ms / 1000 * wf.getsampwidth())

        speech_segments = []
        for i in range(0, len(audio_frames), frame_size):
            frame = audio_frames[i:i + frame_size]
            if len(frame) == frame_size:  # Validate frame size
                is_speech = vad.is_speech(frame, sample_rate=sample_rate)
                if is_speech:
                    speech_segments.append(frame)

    if speech_segments:
        with wave.open(output_audio, 'wb') as wf_out:
            wf_out.setparams(wf.getparams())
            wf_out.writeframes(b"".join(speech_segments))
        print(f"VAD segments saved to {output_audio}")
    else:
        print("No speech detected. Output file not created.")

if __name__ == "__main__":
    input_audio = sys.argv[1]
    output_audio = sys.argv[2]
    vad_detection(input_audio, output_audio)
