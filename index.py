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
    """Generate an audio file from the given text using TTS with a custom file name."""
    # Use a default file name if none is provided, and strip extra spaces.
    file_name_base = file_name_base.strip() if file_name_base else "generated_audio"
    # Generate a unique identifier for the file name.
    unique_id = uuid.uuid4()
    # Create the output file path.
    output_path = f"output/{file_name_base}_{unique_id}.wav"
    
    # Generate the audio file
    tts.tts_to_file(
        text=text,
        file_path=output_path,
        speaker_wav="reference_voices/referance_male.wav",  # Path to your reference voice file
        language="en",         # Change language code if needed
        split_sentences=True,
        speed=1.2
    )
    return output_path

# Create a Gradio interface with two input fields: one for the text and one for the file name base.
interface = gr.Interface(
    fn=generate_audio,
    inputs=[
        gr.Textbox(lines=5, placeholder="Enter your text here...", label="Input Text"),
        gr.Textbox(lines=1, placeholder="Enter file name base...", label="File Name Base")
    ],
    outputs=gr.Audio(type="filepath", label="Generated Audio"),
    title="Text-to-Speech Generator",
    description="Enter your text and a base file name, then generate audio that you can play or download."
)

interface.launch()
