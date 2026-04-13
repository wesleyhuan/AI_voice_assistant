import os
import numpy as np
from faster_whisper import WhisperModel
from config import CFG

# Singleton model — loaded once and reused across calls
_model: WhisperModel | None = None

def get_model(model_size: str = CFG.MODEL_SIZE, device: str = CFG.DEVICE, compute_type: str = CFG.COMPUTE_TYPE) -> WhisperModel:
    """
    Returns a cached WhisperModel, loading it on first call.

    :param model_size: Whisper model size (tiny, base, small, medium, large-v2)
    :param device: Inference device ('cpu' or 'cuda')
    :param compute_type: Numerical precision ('int8', 'int8_float16', 'float16', 'float32')
    :return: Loaded WhisperModel instance.
    """
    global _model
    if _model is None:
        print(f"Loading model '{model_size}' ...")
        _model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("Model ready.")
    return _model

def transcribe_audio(audio_input, model_size: str = CFG.MODEL_SIZE, device: str = CFG.DEVICE, compute_type: str = CFG.COMPUTE_TYPE):
    """
    Transcribes audio to text using faster-whisper.

    Accepts either a file path or a NumPy audio array so that real-time
    callers can skip the save-to-disk round-trip.

    :param audio_input: File path (str) or NumPy float32 array of audio samples.
    :param model_size: Whisper model size (tiny, base, small, medium, large-v2)
    :param device: Inference device ('cpu' or 'cuda')
    :param compute_type: Numerical precision ('int8', 'int8_float16', 'float16', 'float32')
    :return: Full transcribed text as a string, or None on failure.
    """
    # Validate file path input
    if isinstance(audio_input, str) and not os.path.isfile(audio_input):
        raise FileNotFoundError(f"Audio file not found: {audio_input}")

    try:
        model = get_model(model_size, device, compute_type)

        # faster-whisper accepts a 1-D float32 numpy array directly
        if isinstance(audio_input, np.ndarray):
            audio_input = audio_input.flatten().astype(np.float32)

        segments, info = model.transcribe(audio_input, beam_size=5)

        print(f"Detected language: {info.language} (confidence: {info.language_probability:.2f})")

        full_text = []
        for segment in segments:
            full_text.append(segment.text.strip())

        return " ".join(full_text)

    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

if __name__ == "__main__":
    audio_file = CFG.FILE_NAME
    result = transcribe_audio(audio_file)
    if result:
        print("Transcript:", result)
