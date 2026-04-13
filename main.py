import numpy as np
from audio import record_chunk, compute_rms
from audio_text import get_model, transcribe_audio
from config import CFG

def run_realtime():
    """
    Continuously listens to the microphone and transcribes each complete
    utterance as soon as silence is detected after speech.

    Voice Activity Detection uses a simple RMS energy threshold:
      - Energy above CFG.SILENCE_THRESHOLD  → speech
      - CFG.SILENCE_CHUNKS consecutive quiet chunks → utterance ended
    """
    # Pre-load the model so the first utterance isn't delayed
    get_model()
    print("Listening... (Ctrl+C to stop)\n")

    speech_buffer = []
    silence_count = 0
    in_speech = False

    while True:
        chunk = record_chunk()
        rms = compute_rms(chunk)

        if rms > CFG.SILENCE_THRESHOLD:
            if not in_speech:
                print("[Speech detected]")
                in_speech = True
            speech_buffer.append(chunk)
            silence_count = 0

        elif in_speech:
            # Keep buffering briefly so we don't cut off trailing syllables
            speech_buffer.append(chunk)
            silence_count += 1

            if silence_count >= CFG.SILENCE_CHUNKS:
                # Utterance finished — transcribe the accumulated audio
                audio = np.concatenate(speech_buffer, axis=0)
                transcript = transcribe_audio(audio)
                if transcript:
                    print(f"You said: {transcript}\n")
                else:
                    print("[Could not transcribe]\n")

                # Reset for the next utterance
                speech_buffer = []
                silence_count = 0
                in_speech = False

def main():
    try:
        run_realtime()
    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == "__main__":
    main()
