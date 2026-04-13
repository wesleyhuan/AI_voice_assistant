import sounddevice as sd
import soundfile as sf
import numpy as np
from config import CFG

def record_audio(duration=CFG.BLOCK_DURATION, sample_rate=CFG.SAMPLE_RATE):
    """
    Records a fixed-length audio block from the default microphone.

    :param duration: Recording length in seconds.
    :param sample_rate: Sampling rate in Hz.
    :return: NumPy array containing recorded audio data, or None on failure.
    """
    if duration <= 0:
        raise ValueError("Duration must be a positive number.")
    if sample_rate <= 0:
        raise ValueError("Sample rate must be a positive number.")

    print(f"Recording for {duration} s...")
    try:
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()
        print("Recording complete.")
        return audio_data
    except Exception as e:
        print(f"Error during recording: {e}")
        return None

def record_chunk(duration=CFG.CHUNK_DURATION, sample_rate=CFG.SAMPLE_RATE):
    """
    Records a short audio chunk for real-time processing.

    :param duration: Chunk length in seconds.
    :param sample_rate: Sampling rate in Hz.
    :return: NumPy array of shape (samples, 1).
    """
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    return audio_data

def compute_rms(audio_data):
    """
    Computes the root-mean-square energy of an audio chunk.
    Used for simple voice activity detection.

    :param audio_data: NumPy array of audio samples.
    :return: RMS energy as a float.
    """
    return float(np.sqrt(np.mean(audio_data ** 2)))

def play_audio(audio_data, sample_rate=CFG.SAMPLE_RATE):
    """
    Plays audio data through the default output device.

    :param audio_data: NumPy array containing audio samples.
    :param sample_rate: Sampling rate in Hz.
    """
    if audio_data is None or len(audio_data) == 0:
        print("No audio data to play.")
        return

    print("Playing audio...")
    try:
        sd.play(audio_data, samplerate=sample_rate)
        sd.wait()
        print("Playback finished.")
    except Exception as e:
        print(f"Error during playback: {e}")

def save_audio(audio_data, filename=CFG.FILE_NAME, sample_rate=CFG.SAMPLE_RATE):
    """
    Saves audio data to a WAV file.

    :param audio_data: NumPy array containing audio samples.
    :param filename: Output file path.
    :param sample_rate: Sampling rate in Hz.
    :return: The saved filename, or None on failure.
    """
    try:
        sf.write(filename, audio_data, sample_rate)
        print(f"Audio saved to {filename}")
        return filename
    except Exception as e:
        print(f"Error saving audio: {e}")
        return None

if __name__ == "__main__":
    try:
        record = record_audio()
        play_audio(record)
        save_audio(record)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
