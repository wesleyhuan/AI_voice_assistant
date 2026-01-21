import sounddevice as sd
import soundfile as sf
import numpy
from config import CFG

def recoed_audio(duration=CFG.BLOCK_DURATION, sample_rate=CFG.SAMPLE_RATE):
    """
    Records audio from the default microphone
    
    :param duration: Description
    :param sample_rate: Description
    :return Numpy array containing recorded audio data
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
        sd.wait()  # Wait until playback is finished
        print("Playback finished.")
    except Exception as e:
        print(f"Error during playback: {e}")

def save_audio(audio_data, filename=CFG.FILE_NAME, sample_rate = CFG.SAMPLE_RATE):
    sf.write(filename, audio_data, sample_rate)
    print(f"已儲存為 {filename}")
    return filename

if __name__ == "__main__":
    try:
        # record 5 s of audio
        record = recoed_audio()
        # Play back the recorded audio
        play_audio(record)
        # Save audio file
        save_audio(record)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
