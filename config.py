class CFG:
    # === Audio settings ===
    SAMPLE_RATE = 16000       # Whisper recommended 16kHz
    CHANNELS = 1              # Mono
    BLOCK_DURATION = 5        # Seconds for standalone audio.py demo

    # === Real-time / VAD settings ===
    CHUNK_DURATION = 0.5      # Seconds per recorded chunk
    SILENCE_THRESHOLD = 0.01  # RMS energy below this is treated as silence
    SILENCE_CHUNKS = 4        # Consecutive silent chunks before utterance ends (~2 s)

    # === Whisper model settings ===
    MODEL_SIZE = "small"      # tiny, base, small, medium, large-v2
    DEVICE = "cpu"            # 'cpu' or 'cuda'
    COMPUTE_TYPE = "int8"     # 'int8', 'float16', 'float32'

    FILE_NAME = "./output.wav"
