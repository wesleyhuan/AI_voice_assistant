from audio import record_audio, play_audio, save_audio
from audio_text import transcribe_audio
from config import CFG

def main():
    # Step 1: Record audio from microphone
    audio_data = record_audio()
    if audio_data is None:
        print("Recording failed. Exiting.")
        return

    # Step 2: Save the recording to a file
    saved_file = save_audio(audio_data)
    if saved_file is None:
        print("Failed to save audio. Exiting.")
        return

    # Step 3: Transcribe the saved audio file
    transcript = transcribe_audio(saved_file)
    if transcript:
        print("\nFinal transcript:", transcript)
    else:
        print("Transcription failed.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
