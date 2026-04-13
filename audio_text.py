import os
from faster_whisper import WhisperModel
from config import CFG

def transcribe_audio(file_path: str, model_size: str = CFG.MODEL_SIZE, device: str = CFG.DEVICE, compute_type: str = CFG.COMPUTE_TYPE):
    """
    Transcribes an audio file to text using faster-whisper.

    :param file_path: Path to the audio file (supports mp3, wav, m4a, etc.)
    :param model_size: Whisper model size (tiny, base, small, medium, large-v2)
    :param device: Inference device ('cpu' or 'cuda')
    :param compute_type: Numerical precision ('int8', 'int8_float16', 'float16', 'float32')
    :return: Full transcribed text as a string, or None on failure.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        print(f"Loading model '{model_size}' ...")
        model = WhisperModel(model_size, device=device, compute_type=compute_type)

        print("Starting transcription...")
        segments, info = model.transcribe(file_path, beam_size=5)

        print(f"Detected language: {info.language} (confidence: {info.language_probability:.2f})")
        print("Transcription:")

        full_text = []
        for segment in segments:
            print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
            full_text.append(segment.text)

        return " ".join(full_text)

    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

if __name__ == "__main__":
    audio_file = CFG.FILE_NAME
    result = transcribe_audio(audio_file)
    if result:
        print("\nFull transcript:", result)
