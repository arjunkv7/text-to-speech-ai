import gradio as gr
import torch
import uuid  # Import uuid to generate unique IDs
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig, XttsArgs
from TTS.tts.models.xtts import XttsAudioConfig  # Import the missing class
from TTS.config.shared_configs import BaseDatasetConfig

# Set device (GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Allow safe globals so that XttsConfig and XttsAudioConfig are permitted during checkpoint loading
torch.serialization.add_safe_globals([XttsConfig, XttsArgs, XttsAudioConfig, BaseDatasetConfig])

# Initialize the TTS model (using XTTS-v2)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def generate_audio(text):
    """Generate an audio file from the given text using TTS with a unique file name."""
    # Generate a unique file name using uuid
    unique_id = uuid.uuid4()
    output_path = f"output/generated_audio_{unique_id}.wav"
    
    # Generate the audio file
    tts.tts_to_file(
        text=text,
        file_path=output_path,
        speaker_wav="reference_voices/referance_male.wav",  # Path to your reference voice file
        language="en",         # Change language code if needed
        split_sentences=True,
        speed=1
    )
    return output_path

# Create a Gradio interface with a textbox input and an audio output.
interface = gr.Interface(
    fn=generate_audio,
    inputs=gr.Textbox(lines=5, placeholder="Enter your text here...", label="Input Text"),
    outputs=gr.Audio(type="filepath", label="Generated Audio"),
    title="Text-to-Speech Generator",
    description="Enter text, generate audio, and then play or download the resulting audio file."
)

interface.launch()
