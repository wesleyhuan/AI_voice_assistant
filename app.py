import numpy as np
import gradio as gr
from audio_text import get_model, transcribe_audio
from config import CFG

# Pre-load the model so the first recording isn't delayed
get_model()

def handle_audio(audio):
    """
    Gradio handler: receives microphone audio and returns the transcript.

    :param audio: Tuple of (sample_rate, numpy_array) provided by gr.Audio.
    :return: Transcribed text string.
    """
    if audio is None:
        return "", []

    sample_rate, audio_data = audio

    # Gradio provides int16 by default — normalise to float32 for Whisper
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max

    transcript = transcribe_audio(audio_data)
    return transcript if transcript else "[Could not transcribe]"

def update_history(transcript, history):
    """Prepend the latest transcript to the running history list."""
    if transcript and transcript != "[Could not transcribe]":
        history = [transcript] + history
    return history

with gr.Blocks(title="AI Voice Assistant") as demo:
    gr.Markdown("# AI Voice Assistant\nRecord your voice and get an instant transcript.")

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["microphone"],
                type="numpy",
                label="Microphone",
            )
            model_info = gr.Markdown(
                f"**Model:** `{CFG.MODEL_SIZE}` &nbsp;|&nbsp; "
                f"**Device:** `{CFG.DEVICE}` &nbsp;|&nbsp; "
                f"**Precision:** `{CFG.COMPUTE_TYPE}`"
            )

        with gr.Column(scale=1):
            transcript_box = gr.Textbox(
                label="Transcript",
                placeholder="Transcript will appear here...",
                lines=4,
            )
            history_box = gr.JSON(label="History", value=[])

    clear_btn = gr.Button("Clear history")

    # Wire up: when audio changes, transcribe → update transcript + history
    audio_input.change(
        fn=handle_audio,
        inputs=[audio_input],
        outputs=[transcript_box],
    ).then(
        fn=update_history,
        inputs=[transcript_box, history_box],
        outputs=[history_box],
    )

    clear_btn.click(fn=lambda: ([], ""), outputs=[history_box, transcript_box])

if __name__ == "__main__":
    demo.launch()
