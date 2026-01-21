class CFG:
    # === 參數設定 ===
    SAMPLE_RATE = 16000       # Whisper 建議 16kHz
    CHANNELS = 1              # 單聲道
    BLOCK_DURATION = 5        # 每次錄音秒數 (越短越即時，但準確度可能略降)
    MODEL_SIZE = "small"      # 模型大小 (tiny, base, small, medium, large-v2)
    DEVICE = "cpu"            # 'cpu' 或 'cuda'
    COMPUTE_TYPE = "int8"     # 'int8', 'float16', 'float32'
    FILE_NAME = "./output.wav"