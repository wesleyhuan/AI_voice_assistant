import os
from faster_whisper import WhisperModel
from config import CFG

def transcribe_audio(file_path: str, model_size: str = CFG.MODEL_SIZE, device: str = CFG.DEVICE, compute_type: str = CFG.COMPUTE_TYPE):
    """
    使用 faster-whisper 將音訊檔轉換成文字
    :param file_path: 音訊檔路徑 (支援 mp3, wav, m4a 等)
    :param model_size: 模型大小 (tiny, base, small, medium, large-v2)
    :param device: 運行設備 ('cpu' 或 'cuda')
    :param compute_type: 計算精度 ('int8', 'int8_float16', 'float16', 'float32')
    """
    # 檢查檔案是否存在
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"找不到檔案: {file_path}")

    try:
        # 載入模型
        print(f"載入模型 {model_size} ...")
        model = WhisperModel(model_size, device=device, compute_type=compute_type)

        # 開始轉錄
        print("開始轉錄...")
        segments, info = model.transcribe(file_path, beam_size=5)

        print(f"語言偵測結果: {info.language} (信心度: {info.language_probability:.2f})")
        print("轉錄內容：")
        for segment in segments:
            print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

    except Exception as e:
        print(f"轉錄過程發生錯誤: {e}")

audio_file = CFG.FILE_NAME  # 請換成你的檔案路徑
transcribe_audio(audio_file)