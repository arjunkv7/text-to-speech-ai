import os
import uuid
import gradio as gr
import torch
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig, XttsArgs
from TTS.tts.models.xtts import XttsAudioConfig  # Import the missing class
from TTS.config.shared_configs import BaseDatasetConfig

# Ensure the output directory exists
os.makedirs("output", exist_ok=True)

# Set device (GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Allow safe globals so that XttsConfig and XttsAudioConfig are permitted during checkpoint loading
torch.serialization.add_safe_globals([XttsConfig, XttsArgs, XttsAudioConfig, BaseDatasetConfig])

# Initialize the TTS model (using XTTS-v2)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def generate_audio(text, file_name_base):
    """
    Generate an audio file from the given text using TTS and return the file path.
    The file name is created using a base name provided by the user plus a unique UUID.
    """
    file_name_base = file_name_base.strip() if file_name_base else "generated_audio"
    unique_id = uuid.uuid4()
    output_file = f"{file_name_base}_{unique_id}.wav"
    output_path = os.path.join("output", output_file)
    
    tts.tts_to_file(
        text=text,
        file_path=output_path,
        speaker_wav="reference_voices/referance_male.wav",  # Update the path if needed.
        language="en",         # Change language code if needed.
        split_sentences=True,
        speed=1.2
    )
    # Return the output path twice:
    # - Once for the audio widget.
    # - Once to store as state for later deletion.
    return output_path, output_path

def delete_audio(file_path):
    """
    Delete the generated audio file using the provided file path.
    Returns an empty string for the audio output (to clear it) and a status message.
    """
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
            return "", "File deleted successfully."
        except PermissionError as e:
            return file_path, f"Permission error: {str(e)}. Please ensure the file is not in use."
    else:
        return file_path, "File not found or already deleted."

with gr.Blocks() as demo:
    gr.Markdown("# Text-to-Speech Generator")
    
    # Input row for text and file name base.
    with gr.Row():
        text_input = gr.Textbox(lines=5, placeholder="Enter your text here...", label="Input Text")
        file_name_input = gr.Textbox(lines=1, placeholder="Enter file name base...", label="File Name Base")
    
    # Row for buttons.
    with gr.Row():
        generate_button = gr.Button("Generate Audio")
        delete_button = gr.Button("Delete Audio")
    
    # Audio output widget and status display.
    audio_output = gr.Audio(type="filepath", label="Generated Audio")
    status_output = gr.Textbox(label="Status")
    
    # Hidden state to store the current generated file path.
    current_file = gr.State(value="")
    
    # Generate button updates both the audio output and the hidden state.
    generate_button.click(fn=generate_audio, inputs=[text_input, file_name_input],
                            outputs=[audio_output, current_file])
    
    # Delete button uses the stored file path to remove the file and clears the audio output.
    delete_button.click(fn=delete_audio, inputs=current_file,
                        outputs=[audio_output, status_output])

demo.launch()
